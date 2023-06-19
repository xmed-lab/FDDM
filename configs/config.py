from torchvision import transforms


class Config(object):
    modality = "mm"
    net_name = "%s-model" % modality

    cls_num = 11

    # CAM
    heatmap = False
    loosepair = True
    if_syn = False

    # training details
    fine_tuning = False
    if fine_tuning:
        lr = 0.0001
    else:
        lr = 0.001
    checkpoint = ""

    train_params = {
        "optimizer": "sgd",
        "sgd": {
            "lr": lr,
            "lr_decay": 0.5,
            "lr_decay_start":5,
            "tolerance_iter_num": 5,
            "lr_min": 1e-7,
            "momentum": 0.9,
            "weight_decay": 1e-4
        },
        "samples_num": 6400,
        "batch_size": 8,
        "print_freq": 10,
        "max_epoch": 80,
        "best_metric": "accuracy"
    }

    # normalization
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # traditional data augmentation (for train) or just resize (for val or test)
    aug_params = {
        "augmentation": {
            "output_shape": [448, 448],
            "rotation": True, "rotation_range": [0, 360],
            "contrast": True, "contrast_range": [0.7, 1.3],
            "brightness": True, "brightness_range": [0.7, 1.3],
            "color": True, "color_range": [0.7, 1.3],
            "multiple_rgb": False, "multiple_range": [0.7, 1.3],
            "flip": True, "flip_prob": 0.5,
            "crop": True, "crop_prob": 0.4,
            "crop_w": 0.04, "crop_h": 0.05,
            "keep_aspect_ratio": False,
            "resize_pad": False,
            "zoom": True, "zoom_prob": 0.5,
            "zoom_range": [0.00, 0.05],
            "paired_transfos": False,
            "rotation_expand": False,
            "crop_height": False,
            "extra_width_crop": False,
            "extra_height_crop": False,
            "crop_after_rotation": False
        },
        "onlyresize": {
            "output_shape": [448, 448],
            "rotation": False, "rotation_range": [0, 360],
            "contrast": False, "contrast_range": [0.7, 1.3],
            "brightness": False, "brightness_range": [0.7, 1.3],
            "color": False, "color_range": [0.7, 1.3],
            "multiple_rgb": False, "multiple_range": [0.7, 1.3],
            "flip": False, "flip_prob": 0.5,
            "crop": False, "crop_prob": 0.4,
            "crop_w": 0.04, "crop_h": 0.05,
            "keep_aspect_ratio": False,
            "resize_pad": False,
            "zoom": False, "zoom_prob": 0.5,
            "zoom_range": [0.00, 0.05],
            "paired_transfos": False,
            "rotation_expand": False,
            "crop_height": False,
            "extra_width_crop": False,
            "extra_height_crop": False,
            "crop_after_rotation": False
        }
    }


