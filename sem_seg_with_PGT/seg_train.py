from torch import optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from sem_seg_with_PGT.seg_utils import log_preds, calculate_iou
import torch
import segmentation_models_pytorch as smp
import importlib
from cam_generator import GeneratorCAM
import datasets_seg
from class_agnostic_seg.ca_utils import save_model
import os
from segmentation_models_pytorch.losses import FocalLoss


def train(train_loader, model, optimizer, loss_fn, args, scaler=None):
    model.train()
    loop = tqdm(train_loader)

    for batch_idx, data in enumerate(loop):
        inputs, targets, _, _ = data

        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        targets = targets.squeeze(1)

        if args.device == "cuda":
            with torch.cuda.amp.autocast():
                predictions = model(inputs)
                loss = loss_fn(predictions, targets.to(torch.int64))
        else:
            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

        # backward
        if args.log_wandb:
            wandb.log({"train_loss": loss.item()})
        optimizer.zero_grad()

        if args.device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        else:
            loss.backward()
            optimizer.step()

        loop.set_postfix(loss=loss.item())


def run(args):
    # initialize classifier for CAM generation
    classification_model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    classification_model.load_state_dict(torch.load(args.cam_weights_name, map_location=args.device), strict=True)
    classification_model.to(args.device)
    classification_model.eval()
    args.cam_generator = GeneratorCAM(model=classification_model, candidate_layers=[args.target_layer])

    # initialize class agnostic model for generating pseudo masks
    class_agnostic_model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, in_channels=5, classes=1).to(args.device)
    checkpoint = torch.load(os.path.join(args.save_model_path,  args.class_agnostic_model_name),
                            map_location=torch.device(args.device))
    class_agnostic_model.load_state_dict(checkpoint["state_dict"])
    args.class_agnostic_model = class_agnostic_model
    args.class_agnostic_model_name = args.class_agnostic_model_name.split(".")[0]

    # prepare data
    train_ds, val_ds = datasets_seg.get_pseudo(args)
    train_loader = DataLoader(dataset=train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=1, shuffle=False)

    # initialize segmentation model, loss, optimizer
    model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, in_channels=3, classes=21).to(args.device)
    loss_fn = FocalLoss(mode="multiclass")
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler() if args.device == "cuda" else None

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
                "class_agnostic_model_name": args.class_agnostic_model_name,
            }
        )
        args.save_model_name = f"{args.data_cls_sub}_{wandb.run.name}.pth.tar"

    calculate_iou(val_ds, model, device=args.device, log=args.log_wandb)
    best_val_miou = 0

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        train(train_loader, model, optimizer, loss_fn, args, scaler)

        iou, miou = calculate_iou(val_ds, model, device=args.device, log=args.log_wandb)

        if miou > best_val_miou:
            best_val_miou = miou
            save_model(model, optimizer, args)

        if args.log_wandb:
            log_preds(val_loader, model, device=args.device, batches=[x for x in range(10)], val=True)
            log_preds(train_loader, model, device=args.device, batches=[0, 1], val=False)

    if args.log_wandb:
        wandb.finish()
