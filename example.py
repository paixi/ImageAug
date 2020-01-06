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
