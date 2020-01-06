# ImageAug
Image augmentation for PyTorch

* Apply random cropped rotations without going out of image bounds
* Convert RGB to YUV color space
* Adjust brightness and contrast, and more

![Example](https://i.imgur.com/lyINe9Z.gif "Example")

[Artwork](https://twitter.com/hcnone/status/1085740161600651269) by @[hcnone](https://twitter.com/hcnone)

## Quick Start

The transformations are designed to be chained together using `torchvision.transforms.Compose`. Additionally, there is a functional module. Functional transforms give more fine-grained control if you have to build a more complex transformation pipeline.

### Install

```sh
pip3 install -r requirements.txt
python3 setup.py install
```
**Requirements:**

* Pillow
* torchvision
* numpy

### Example

```python
from torchvision.transforms import ToTensor, ToPILImage, Compose
from PIL import Image
from imageaug.transforms import Colorspace, RandomAdjustment, RandomRotatedCrop

image_filename = 'test.png'
img = Image.open(image_filename, 'r').convert("RGB")

crop_size = (64, 64)
angle_std = 90 # in degrees
# Note: apply color adjustments before a random rotated crop so that so that the
#       fillcolor for out of bounds is not randomly adjusted (this only applies
#       if you have images smaller than the crop size)
transform = Compose([
    # convert PIL Image to Tensor
    ToTensor(),
    # convert RGB to YUV colorspace
    Colorspace("rgb", "yuv"),
    # randomly adjust the brightness and contrast of channel 0 (Y: luminance)
    RandomAdjustment(0, 0.1, 0.1, rgb=False),
    # randomly adjust the contrast of channel 1 and 2 (UV: color channels)
    RandomAdjustment((1,2), 0, 0.38, rgb=False),
    # convert YUV to RGB colorspace
    Colorspace("yuv", "rgb"),
    # convert Tensor back to PIL Image
    ToPILImage(),
    # random rotated crop
    RandomRotatedCrop(crop_size, 0.0, angle_std, downscale=0.5)
])
out = transform(img)
out.save("out.png")
```

## Current Features
* Rotate and crop images within the bounds of the original image for any given degree of angle perturbation (for training samples with rotational noise)
* Convert images to and from RGB/YUV/YCH colorspace with alpha channel support
* Adjust contrast and brightness of channels
* Noise occulsion

## To-do

This project is still a work in progress.

* Uniform distribution for RandomRotatedCrop
* Color lookup table for faster conversions between colorspaces
* Add image, text, shape, and pixelation occulsions

## Project Page
Github: [https://github.com/paixi/ImageAug](https://github.com/paixi/ImageAug)