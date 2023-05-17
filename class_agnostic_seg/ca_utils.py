import os
import torch
from chainercv.evaluations import calc_semantic_segmentation_confusion
import numpy as np
import wandb
from tqdm import tqdm
from voc import CATS_DICT


def save_checkpoint(state, folder=".", filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, os.path.join(folder, filename))


def save_model(model, optimizer, args):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    save_checkpoint(checkpoint, folder=args.save_model_path,
                    filename=args.save_model_name)
    print("=> saved best model")


def load_checkpoint(checkpoint, model):
    print("=> Saving checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_inputs_labels(data, mode):
    if mode == "both":
        inputs, labels, _, _, _, _ = data
    elif mode == "cam":
        _, labels, inputs, _, _ = data
    elif mode == "img":
        inputs, labels, _, _ = data
    else:
        raise ValueError("Unknown mode")
    return inputs, labels


def get_input_channels(mode, negative_cams):
    if mode == "both":
        if negative_cams:
            input_channels = 5
        else:
            input_channels = 4
    elif mode == "cam":
        input_channels = 1
    elif mode == "img":
        input_channels = 3
    else:
        raise ValueError("Unknown mode")
    return input_channels


def batch_loss(loader, model, loss_fn, device, mode, log=False, image_set="val"):
    valid_loss = 0
    model.eval()
    with torch.no_grad():

        for data in loader:
            inputs, labels = get_inputs_labels(data, mode)

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            valid_loss += loss.item()

    if log:
        wandb.log({f"{image_set} loss": valid_loss})
    return valid_loss


def calculate_iou(ds, model, device, threshold=0.5, num_classes=21, binary=False, log=False):
    """
    Calculate IoU for each class and mIoU
    Args:
    ds: dataset
    model: model to evaluate
    device: device to use
    threshold: threshold for prediction
    num_classes: number of classes + background
    binary: if True, calculate IoU for binary segmentation (foreground vs background)
    log: if True, log results to wandb
    Returns:
    iou: IoU for each class
    miou: mIoU for all classes except background
    """
    model.eval()
    loop = tqdm(ds)
    confusion = np.zeros((num_classes, num_classes))
    with torch.no_grad():
        for i, data in enumerate(loop):
            inputs, mask, _, _, targets, _ = data
            inputs = torch.from_numpy(inputs).float()

            inputs = inputs.to(device)
            preds = model(torch.unsqueeze(inputs, dim=0))
            preds = torch.squeeze(preds, dim=0)

            preds = (preds > threshold).float()
            cls_idx = torch.argmax(targets, dim=0)

            preds.reshape([preds.size(-2), preds.size(-1)])
            mask.reshape([mask.size(-2), mask.size(-1)])

            preds = preds.detach().cpu().numpy().astype(np.int32)
            mask = mask.numpy().astype(np.int32)

            if not binary:
                preds = np.where(preds == 1, cls_idx, num_classes - 1)
                mask = np.where(mask == 1, cls_idx, num_classes - 1)

            confusion += calc_semantic_segmentation_confusion(preds, mask)[:num_classes, :num_classes]

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator
    miou = np.nanmean(iou[:num_classes - 1])

    if binary:
        iou_d = {"background": round(iou[0], 3) * 100, "foreground": round(iou[1], 3) * 100}
        miou = round(iou[1], 3) * 100

    else:
        iou_d = {CATS_DICT[i + 1]: round(iou[i], 3) * 100 for i in range(num_classes - 1)}
        iou_d["background"] = round(iou[num_classes - 1], 3) * 100

    if log:
        wandb.log({"iou": miou})
    return iou_d, miou


def log_imgs_masks_preds(val_loader, model, args, batches=[1], threshold=0.5, val=True):
    """
    Log images, masks and predictions to wandb
    Args:
    val_loader: validation loader
    model: model to evaluate
    args: arguments
    batches: batches to log
    threshold: threshold for prediction
    val: if True, log as validation set, else as train set
    """
    model.eval()
    mode = args.mode

    with torch.no_grad():
        for b_idx, data in enumerate(val_loader):
            if b_idx not in batches:
                continue
            images_to_log = []
            if mode == "both" or mode == "cam":
                imgs, masks, cam, other_cam, _, _ = data
            elif mode == "img":
                imgs, masks, _, _ = data
            else:
                raise ValueError("Unknown mode")

            if mode == "cam":
                inputs = cam.clone()

            else:
                inputs = imgs.clone()

            inputs, masks = inputs.to(args.device, dtype=torch.float), masks.to(args.device, dtype=torch.float)
            preds = model(inputs.to(args.device))
            preds = torch.sigmoid(preds)
            bin_preds = (preds > threshold).float()
            n = preds.size(0)
            n_vis = min(n, 15)

            for i in range(n_vis):
                images_to_log.append(wandb.Image(imgs[i, :3].cpu().permute([1, 2, 0]).numpy()))
                if mode != "img":
                    images_to_log.append(wandb.Image(torch.cat([cam[i], cam[i], cam[i]], dim=0)))
                if mode == "both":
                    images_to_log.append(wandb.Image(torch.cat([other_cam[i], other_cam[i], other_cam[i]], dim=0)))
                images_to_log.append(wandb.Image(torch.cat([masks[i], masks[i], masks[i]], dim=0)))
                images_to_log.append(wandb.Image(torch.cat([preds[i], preds[i], preds[i]], dim=0)))
                images_to_log.append(wandb.Image(torch.cat([bin_preds[i], bin_preds[i], bin_preds[i]], dim=0)))

            image_set = "val" if val else "train"
            wandb.log({f"examples_{image_set}_{b_idx}": images_to_log})
