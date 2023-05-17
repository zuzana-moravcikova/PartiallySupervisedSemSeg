import importlib
import os.path
import torch

import datasets_seg
from class_agnostic_seg.ca_utils import calculate_iou

from cam_generator import GeneratorCAM
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

def run(args):
    # prepare classifier for CAM generation
    classification_model = getattr(importlib.import_module(args.cam_network), 'CAM')()
    classification_model.load_state_dict(torch.load(args.cam_weights_name, map_location=args.device), strict=True)
    classification_model.to(args.device)
    classification_model.eval()
    args.cam_generator = GeneratorCAM(model=classification_model, candidate_layers=[args.target_layer])

    if args.mode == "img":
        IN_CHANNELS = 3
    elif args.mode == "cam":
        IN_CHANNELS = 1
    elif args.mode == "both":
        IN_CHANNELS = 4
        if args.other_cams:
            IN_CHANNELS = 5
    else:
        raise ValueError("Wrong mode")

    model = smp.DeepLabV3Plus(encoder_name=args.encoder_name, in_channels=IN_CHANNELS, classes=1).to(args.device)
    checkpoint = torch.load(os.path.join(args.save_model_path, args.class_agnostic_model_name), map_location=torch.device(args.device))
    model.load_state_dict(checkpoint["state_dict"])


    val_dataset = datasets_seg.get_sub_bi_cam_one(args)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                           num_workers=args.num_workers)


    iou, miou = calculate_iou(val_dataset, model, args.device)
    print("iou: ", iou)
    print("miou: ", miou)
    bin_iou, bin_miou = calculate_iou(val_dataset, model, args.device, binary=True, num_classes=2)
    print("bin iou: ", bin_iou)
    print("bin miou: ", bin_miou)


