import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def get_transforms(train=True):
    if train == True:
        return A.Compose(
            [
                A.Resize(height=512, width=512, p=1),
                A.OneOf(
                    [
                        A.HueSaturationValue(
                            hue_shift_limit=0.1,
                            sat_shift_limit=0.1,
                            val_shift_limit=0.1,
                            p=0.7,
                        ),
                        A.RandomBrightnessContrast(
                            brightness_limit=0.10,
                            contrast_limit=0.10,
                            p=0.9
                        ),
                    ],
                    p=0.7,
                ),
                A.Cutout(
                    num_holes=8, max_h_size=24, max_w_size=24, fill_value=0, p=0.2
                ),
                A.MotionBlur(blur_limit=(3, 5), p=0),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2(p=1.0),
            ]
        )
    else:
        return A.Compose([A.Resize(height=512, width=512, p=1), ToTensorV2(p=1.0)])
