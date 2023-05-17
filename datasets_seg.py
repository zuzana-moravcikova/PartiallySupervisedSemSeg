from torchvision import disable_beta_transforms_warning

disable_beta_transforms_warning()
from transforms import (get_test_transform_for_cls_agnostic,
                        get_train_transform_for_cls_agnostic,
                        get_train_focused_transform,
                        get_train_transform_for_pseudo,
                        get_test_transform_for_pseudo,
                        get_test_focused_transform,
                        get_image_transform_for_classifier)
from voc import (VOCSegmentationSubFgBg,
                 HBBoxTransform,
                 VOCSegmentationSubFgBgCam,
                 VOCSegmentationSubBiCam,
                 SegmentationPseudo,
                 VOCSegmentationSubBi)

def get_voc_seg(args):
    """Get VOC dataset for segmentation"""
    dataset = VOCSegmentationSubBi(root=args.dataset_root,
                                   sub="all",
                                   transform=get_image_transform_for_classifier(),
                                   image_set="trainval")
    return dataset


def get_focused(args):
    """Get train and val fg/bg dataset for segmentation """
    bbox_trans = HBBoxTransform(range=args.bbox_trans_range)
    dataset_train = VOCSegmentationSubFgBg(root=args.dataset_root,
                                           sub=args.data_cls_sub,
                                           transform=get_train_focused_transform(args),
                                           bbox_transform=bbox_trans,
                                           image_set="train")

    dataset_val = VOCSegmentationSubFgBg(root=args.dataset_root,
                                         sub=args.data_cls_sub,
                                         transform=get_test_focused_transform(args),
                                         bbox_transform=bbox_trans,
                                         image_set="val")

    return dataset_train, dataset_val


def get_focused_one(args):
    """Get val fg/bg dataset for segmentation """
    bbox_trans = HBBoxTransform(range=args.bbox_trans_range)
    dataset = VOCSegmentationSubFgBg(root=args.dataset_root,
                                     sub=args.data_cls_sub,
                                     transform=get_test_focused_transform(args),
                                     bbox_transform=bbox_trans,
                                     image_set="val")

    return dataset


def get_focused_with_cam(args):
    """Get train and val fg/bg dataset for segmentation where input consists of CAM or both image and CAM"""
    bbox_trans = HBBoxTransform(range=args.bbox_trans_range)
    dataset_train = VOCSegmentationSubFgBgCam(root=args.dataset_root,
                                              sub=args.data_cls_sub,
                                              transform=get_train_focused_transform(args),
                                              bbox_transform=bbox_trans,
                                              image_set="train",
                                              mode=args.mode,
                                              cam_generator=args.cam_generator,
                                              device=args.device)

    dataset_val = VOCSegmentationSubFgBgCam(root=args.dataset_root,
                                            sub=args.data_cls_sub,
                                            transform=get_test_focused_transform(args),
                                            bbox_transform=bbox_trans,
                                            image_set="val",
                                            mode=args.mode,
                                            cam_generator=args.cam_generator,
                                            device=args.device)

    return dataset_train, dataset_val


def get_focused_with_cam_one(args):
    """Get val fg/bg dataset for segmentation where input consists of CAM or both image and CAM"""
    bbox_trans = HBBoxTransform(range=args.bbox_trans_range)
    dataset = VOCSegmentationSubFgBgCam(root=args.dataset_root,
                                        sub=args.data_cls_sub,
                                        transform=get_test_focused_transform(args),
                                        bbox_transform=bbox_trans,
                                        image_set="val",
                                        mode=args.mode,
                                        cam_generator=args.cam_generator,
                                        device=args.device)
    return dataset


def get_sub_bi_cam(args):
    """Get train and val sub/bi dataset for segmentation
     where input consists image, CAM and aktivation map for other class"""
    train_dataset = VOCSegmentationSubBiCam(root=args.dataset_root,
                                            image_set="train",
                                            joint_transforms=get_train_transform_for_cls_agnostic(args),
                                            mode=args.mode,
                                            cam_generator=args.cam_generator,
                                            device=args.device,
                                            sub=args.data_cls_sub,
                                            other_cams=args.other_cams,
                                            other_cams_max=args.other_cams_max,
                                            other_cams_present=args.other_cams_present)

    val_dataset = VOCSegmentationSubBiCam(root=args.dataset_root,
                                          image_set="val",
                                          joint_transforms=get_test_transform_for_cls_agnostic(args),
                                          mode=args.mode,
                                          cam_generator=args.cam_generator,
                                          device=args.device,
                                          sub=args.data_cls_sub,
                                          other_cams=args.other_cams,
                                          other_cams_max=args.other_cams_max,
                                          other_cams_present=args.other_cams_present)
    return train_dataset, val_dataset

def get_sub_bi_cam_one(args):
    """Get val sub/bi dataset for segmentation
    where input consists image, CAM and aktivation map for other class"""
    dataset = VOCSegmentationSubBiCam(root=args.dataset_root,
                                      image_set="val",
                                      joint_transforms=get_test_transform_for_cls_agnostic(args),
                                      mode=args.mode,
                                      cam_generator=args.cam_generator,
                                      device=args.device,
                                      sub=args.data_cls_sub,
                                      other_cams=args.other_cams,
                                      other_cams_max=args.other_cams_max,
                                      other_cams_present=args.other_cams_present)

    return dataset


def get_pseudo(args):
    """Get train and val dataset for segmentation with pseudo-mask"""
    train_ds = SegmentationPseudo(root=args.dataset_root,
                                  image_set="train",
                                  cam_generator=args.cam_generator,
                                  device=args.device,
                                  pseudo_mask_transform=get_test_transform_for_cls_agnostic(args),
                                  negative_cams=args.other_cams,
                                  class_agnostic_model=args.class_agnostic_model,
                                  transform=get_train_transform_for_pseudo(args),
                                  class_agnostic_model_name=args.class_agnostic_model_name
                                  )
    val_ds = SegmentationPseudo(root=args.dataset_root,
                                image_set="val",
                                cam_generator=args.cam_generator,
                                device=args.device,
                                pseudo_mask_transform=get_test_transform_for_cls_agnostic(args),
                                negative_cams=args.other_cams,
                                class_agnostic_model=args.class_agnostic_model,
                                transform=get_test_transform_for_pseudo(args),
                                class_agnostic_model_name=args.class_agnostic_model_name
                                )

    return train_ds, val_ds
