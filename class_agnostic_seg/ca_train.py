import importlib
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
from class_agnostic_seg.ca_utils import (log_imgs_masks_preds,
                                         batch_loss,
                                         save_model,
                                         get_inputs_labels,
                                         get_input_channels,
                                         calculate_iou
                                         )

import segmentation_models_pytorch as smp
import datasets_seg
from cam_generator import GeneratorCAM


def train(train_loader, model, optimizer, loss_fn, args, scaler=None):
    loop = tqdm(train_loader)
    model.train()

    for batch_idx, data in enumerate(loop):
        inputs, targets = get_inputs_labels(data, args.mode)

        inputs = inputs.to(device=args.device)
        targets = targets.to(device=args.device)

        # forward
        if args.device == "cuda":
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                loss = loss_fn(predictions, targets)
        else:
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()

        if args.device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        if args.log_wandb:
            wandb.log({"loss": loss.item()})
        loop.set_postfix(loss=loss.item())


def run(args):
    if args.log_wandb:
        wandb.init(
            project="class_agnostic_segmentation",
            config={
                "learning_rate": args.learning_rate,
                "architecture": args.model_name,
                "backbone": args.encoder_name,
                "epochs": args.num_epochs,
                "input_size": args.input_size,
                "data_cls_sub": args.data_cls_sub,
                "batch_size": args.batch_size,
                "other_cams": args.other_cams,
                "other_cams_max": args.other_cams_max,
                "other_cams_present": args.other_cams_present,
            }
        )

        # setting up how the model will be saved based on the run name
        args.save_model_name = f"{args.data_cls_sub}_{wandb.run.name}.pth.tar"
        args.run_id = wandb.run.id

    # prepare model for generating CAMs
    classification_model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    classification_model.load_state_dict(torch.load(args.cam_weights_name, map_location=args.device), strict=True)
    classification_model.to(args.device)
    classification_model.eval()
    args.cam_generator = GeneratorCAM(model=classification_model, candidate_layers=[args.target_layer])
    print("=> model for CAMs generation prepared")

    # model initialization
    IN_CHANNELS = get_input_channels(args.mode, args.other_cams)
    OUT_CHANNELS = 1
    model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, in_channels=IN_CHANNELS, classes=OUT_CHANNELS).to(
        args.device)
    # loss and optimizer initialization
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler() if args.device == "cuda" else None

    # data loading
    train_ds, val_ds = datasets_seg.get_sub_bi_cam(args)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=1, shuffle=False)

    if args.log_wandb:
        log_imgs_masks_preds(val_loader, model, args, batches=[x for x in range(25)], val=True)
    calculate_iou(val_ds, model, args.device, binary=True, num_classes=2, log=args.log_wandb)

    best_val_iou = 0
    # training loop
    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")

        train(train_loader, model, optimizer, loss_fn, args, scaler)

        # log validation metrics and images
        _, val_iou = calculate_iou(val_ds, model, args.device, binary=True, num_classes=2, log=args.log_wandb)
        batch_loss(val_loader, model, loss_fn, device=args.device, mode=args.mode, log=args.log_wandb, image_set="val")
        if args.log_wandb:
            log_imgs_masks_preds(val_loader, model, args, batches=[x for x in range(25)], val=True)
            log_imgs_masks_preds(train_loader, model, args, batches=[1, 2], val=False)

        # save model if validation iou is better than previous best
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_model(model, optimizer, args)

    if args.log_wandb:
        wandb.finish()
