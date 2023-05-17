import importlib
import os

import torch
import segmentation_models_pytorch as smp
from cam_generator import GeneratorCAM
import datasets_seg
from sem_seg_with_PGT.seg_utils import calculate_iou


def run(args):
    # initialize classifier for CAM generation
    classification_model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    classification_model.load_state_dict(torch.load(args.cam_weights_name, map_location=args.device), strict=True)
    classification_model.to(args.device)
    classification_model.eval()
    print("=>model for cams loaded")
    args.cam_generator = GeneratorCAM(model=classification_model, candidate_layers=[args.target_layer])

    # initialize class-agnostic model for pseudo mask generation
    args.class_agnostic_model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, in_channels=5, classes=1).to(args.device)
    checkpoint = torch.load(os.path.join(args.save_model_path, args.class_agnostic_model_name),
                            map_location=torch.device(args.device))
    args.class_agnostic_model.load_state_dict(checkpoint["state_dict"])

    # initialize semantic segmentation model for evaluation
    model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, in_channels=3, classes=21).to(args.device)
    checkpoint = torch.load(os.path.join(args.save_model_path, args.pseudo_model_name), map_location=torch.device(args.device))
    model.load_state_dict(checkpoint["state_dict"])
    args.class_agnostic_model_name = args.class_agnostic_model_name.split(".")[0]

    # get dataset
    train_ds, val_ds = datasets_seg.get_pseudo(args)

    val_iou, val_miou = calculate_iou(val_ds, model, device=args.device, pseudo_mask=False)
    pseudo_mask_val_iou, pseudo_mask_val_miou = calculate_iou(val_ds, model, device=args.device, pseudo_mask=True)

    print(f"miou: val: {val_miou}%.2f  |  pseudo_val: {pseudo_mask_val_miou}%.2f")
    print("_____________________________________________")
    print("VAL")
    for key in val_iou:
        print(f" {key}:  {val_iou[key]}%.2f")

    print("_____________________________________________")
    print("PSEUDO_mask-VAL")
    for key in val_iou:
        print(f" {key}:  {pseudo_mask_val_iou[key]}%.2f")
