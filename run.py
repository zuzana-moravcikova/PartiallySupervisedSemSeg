import argparse
from torchvision import disable_beta_transforms_warning
import torch
disable_beta_transforms_warning()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=device, type=str)

    # args for preparing data
    parser.add_argument("--dataset_root", default="./data/VOC", type=str)
    parser.add_argument("--data_cls_sub", default="A", type=str)
    parser.add_argument("--other_cams", default=True, type=bool)
    parser.add_argument("--other_cams_max", default=True, type=bool)
    parser.add_argument("--other_cams_present", default=True, type=int)
    parser.add_argument("--input_size", default=320, type=int)
    # args for focused segmentation
    parser.add_argument("--focused", default=False, type=bool)
    parser.add_argument("--mode", default="both", type=str)
    parser.add_argument("--bbox_trans_range", default=(-0.1, 1.1), type=tuple)
    # hyper-parameters for training
    parser.add_argument("--learning_rate", default=0.0001, type=float)
    parser.add_argument("--num_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    # segmentation model
    parser.add_argument("--model_name", default="DeepLabV3Plus", type=str)
    parser.add_argument("--encoder_name", default="tu-xception71", type=str)
    # CAM generator
    parser.add_argument("--cam_weights_name", default="./ckpts/res50_cam.pth", type=str)
    parser.add_argument("--cam_network", default="classification_model", type=str)
    parser.add_argument("--target_layer", default="stage4", type=str)
    parser.add_argument("--cam_generator", default=None)
    # for loading and saving checkpoints
    parser.add_argument("--save_model", default=True, type=bool)
    parser.add_argument("--save_model_path", default="./ckpts", type=str)
    parser.add_argument("--save_model_name", default="model.pth", type=str)
    parser.add_argument("--run_id", default=0, type=int)
    # for training with pseudo masks
    parser.add_argument("--class_agnostic_model", default=None, type=str)
    parser.add_argument("--class_agnostic_model_name", default="class_agnostic_model.pth", type=str)
    parser.add_argument("--pseudo_model_name", default="semantic_model.pth", type=str)
    # the task
    parser.add_argument("--log_wandb", default=True, type=bool)
    parser.add_argument("--task", default="class_agnostic_train", type=str)
    args = parser.parse_args()

    if args.task == "class_agnostic_train":
        from class_agnostic_seg import ca_train
        ca_train.run(args)

    elif args.task == "class_agnostic_eval":
        from class_agnostic_seg import ca_eval
        assert args.class_agnostic_model_name is not None  # the filename of pre-trained CA model that will be evaluated
        ca_eval.run(args)

    elif args.task == "pseudo_train":
        from sem_seg_with_PGT import seg_train
        assert args.class_agnostic_model_name is not None   # the filename of pre-trained CA model, used for generating pseudo masks
        seg_train.run(args)

    elif args.task == "pseudo_eval":
        from sem_seg_with_PGT import seg_eval
        assert args.class_agnostic_model_name is not None  # the filename of pre-trained model, used for generating pseudo masks
        assert args.pseudo_model_name is not None          # the filename of sem. seg. model that will be evaluated
        seg_eval.run(args)


