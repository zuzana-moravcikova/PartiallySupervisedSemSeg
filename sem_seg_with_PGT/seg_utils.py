import numpy as np
import torch
from tqdm import tqdm
from chainercv.evaluations import calc_semantic_segmentation_confusion
import wandb
from voc import COLORMAP, CATS_DICT


def masks_to_colored_masks(preds, num_classes=21):
    """ Converts 1D predicted semantic segmentation masks to 3D colored masks

    Args:
    preds: predicted semantic segmentation masks
    num_classes: number of classes

    Returns:
    mask: colored masks
    """
    mask_r = torch.zeros(preds.size(0), 1, preds.size(2), preds.size(3))
    mask_g = torch.zeros(preds.size(0), 1, preds.size(2), preds.size(3))
    mask_b = torch.zeros(preds.size(0), 1, preds.size(2), preds.size(3))

    for c in range(preds.size(0)):
        for i in range(num_classes):
            # set mask to 1 for pixels that belong to class i
            mask_r[c, 0, :, :] += (preds[c, 0] == i).float() * COLORMAP[i][0]
            mask_g[c, 0, :, :] += (preds[c, 0] == i).float() * COLORMAP[i][1]
            mask_b[c, 0, :, :] += (preds[c, 0] == i).float() * COLORMAP[i][2]

    mask = torch.cat([mask_r, mask_g, mask_b], dim=1)
    return mask


def calculate_iou(ds, model, device, num_classes=21, pseudo_mask=False, log=False):
    """
    Calculates IoU for each class and mean IoU

    Args:
    ds: dataset
    model: model to evaluate
    device: device to run on
    num_classes: number of classes + background
    pseudo_mask: if True, use pseudo mask instead of model prediction
    log: if True, log images to wandb

    Returns:
    iou_d: dictionary of IoU for each class
    miou: mean IoU of all classes except background
    """
    model.eval()
    loop = tqdm(ds)
    confusion = np.zeros((num_classes, num_classes))

    with torch.no_grad():
        for i, data in enumerate(loop):
            inputs, pseudo, gt, _, _ = data
            inputs = inputs.to(device)

            if pseudo_mask:
                preds = pseudo

            else:
                preds = model(inputs.unsqueeze(0))
                preds = torch.softmax(preds, dim=1)
                preds = torch.argmax(preds, dim=1).detach().cpu()

            preds.reshape([preds.size(-2), preds.size(-1)])
            gt.reshape([gt.size(-2), gt.size(-1)])

            preds = preds.numpy().astype(np.int32) - 1
            gt = gt.numpy().astype(np.int32) - 1

            preds = np.where(preds == -1, num_classes - 1, preds)
            gt = np.where(gt == -1, num_classes - 1, gt)

            confusion += calc_semantic_segmentation_confusion(preds, gt)[:21, :21]


    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    miou = np.nanmean(iou[:num_classes - 1])
    print("miou = ", miou)
    iou_d = {CATS_DICT[i + 1]: round(iou[i], 3) * 100 for i in range(num_classes - 1)}
    iou_d["background"] = round(iou[num_classes - 1], 3) * 100
    if log:
        wandb.log({"miou": miou})
        wandb.log(iou_d)
    return iou_d, miou


def log_preds(loader, model, device, batches=[0, 1], val=True):
    """
    Logs images to wandb
    Args:
    loader: dataloader
    model: model to use
    device: device to run on
    batches: list of batches to log
    val: if True, log ground truth masks
    """
    model.eval()
    with torch.no_grad():

        for i, data in enumerate(loader):
            if i not in batches:
                continue
            images_to_log = []
            if val:
                inputs, pseudo, gt, _, _ = data
            else:
                inputs, pseudo, _, _ = data
            inputs = inputs.to(device)

            preds = model(inputs)
            preds = torch.argmax(preds, dim=1)
            preds = torch.unsqueeze(preds, dim=1)

            if val:
                gt_mask = masks_to_colored_masks(gt.cpu())
            preds_mask = masks_to_colored_masks(preds.cpu())
            pseudo_mask = masks_to_colored_masks(pseudo.cpu())

            for j in range(inputs.shape[0]):
                images_to_log.append(wandb.Image(inputs[j].cpu().permute([1, 2, 0]).numpy()))
                images_to_log.append(wandb.Image((pseudo_mask[j].cpu().permute([1, 2, 0]).numpy())))
                if val:
                    images_to_log.append(wandb.Image(gt_mask[j].cpu().permute([1, 2, 0]).numpy()))
                images_to_log.append(wandb.Image(preds_mask[j].cpu().permute([1, 2, 0]).numpy()))

            image_set = "val" if val else "train"
            wandb.log({f"examples_{image_set}_{i}": images_to_log})
