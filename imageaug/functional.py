'''functional.py - image augmentation functions'''
import random
from math import sin, cos, pi

import torch
import torch.nn.functional as F
import PIL.Image

def pixelate(x, size, area=None):
    '''pixelate area - not implemented '''
    raise NotImplemented
    return x

def rotated_crop_scale(img, crop_size):
    return min(img.width, img.height) / (crop_size[0]**2 + crop_size[1]**2)**0.5

def center(img):
    return img.width/2, img.height/2

def rotated_crop_distance(img, angle, crop_size):
    '''get maximum distance from center of image for a given angle and crop_size
    that is within the bounds of the original unrotated image
    
    Parameters:
        img : PIL.Image
            image to be rotated
        angle: float
            image rotation angle
        crop_size: (int, int)
            width and height of crop area
    
    Returns: (float, float)
        values less than 0 are out of bounds
    '''
    cw, ch = crop_size
    angle = angle/180.*pi
    cw_r = abs(cw * sin(pi/2-angle)) + abs(ch * sin(angle))
    ch_r = abs(ch * sin(pi/2-angle)) + abs(cw * sin(angle))
    dw = img.width/2. - cw_r/2
    dh = img.height/2. - ch_r/2
    return dw, dh 

def random_rotated_crop(img, crop_size, mean, std, downscale=0, resample=PIL.Image.NEAREST, fillcolor=None):
    '''randomly rotate and crop an image within its original boundaries with a
    gaussian distribution
    
    Parameters:
        img : PIL.Image
            image to be rotated
        crop_size: (int, int)
            width and height of crop area
        mean: float
            mean angle (in degrees)
        std: float
            standard deviation (in degrees)
        downscale: float (from 0.0 to 1.0, default: 0)
            controls the scaling before cropping a image
            
            when downscale is 1 or more, the image will be downscaled to fit
            as much of the image into the rotated crop as possible
            
            otherwise, downscale controls how much downscaling to apply
            0.5 will crop a quarter of the image (half the height and width)
            
            set to 0 to always crop the original image without downscaling
        resample: int (default: PIL.Image.BILINEAR)
            PIL.Image resampling filter
            
            possible options as of PIL 6.2.1:
                NEAREST, BILINEAR, BICUBIC
        fillcolor: tuple (int, int, int)
            RGB fill color for out of bound areas when the crop size is
            larger than the image
    Returns: (float, float)
        values less than 0 are out of bounds
    '''
    angle = random.normalvariate(mean, std)
    if downscale > 0:
        crop_scale = rotated_crop_scale(img, crop_size)
        scale = max(1, crop_scale * downscale)
        if resample == PIL.Image.NEAREST:
            img = img.resize((int(round(img.width / scale)), int(round(img.height / scale))), resample=PIL.Image.NEAREST)
        else: # Hamming resampling provides crisper rescaling
            img = img.resize((int(round(img.width / scale)), int(round(img.height / scale))), resample=PIL.Image.HAMMING)
    dw, dh = rotated_crop_distance(img, angle, crop_size)
    img_cx, img_cy = center(img)
    cx = max(0, dw) * (2*random.random()-1)
    cy = max(0, dh) * (2*random.random()-1)
    out = img.rotate(angle, resample=resample, expand=True, fillcolor=fillcolor)
    rx = out.width/2
    ry = out.height/2
    dx = cx * cos(angle/180.*pi) + cy * sin(angle/180.*pi)
    dy = cy * cos(angle/180.*pi) - cx * sin(angle/180.*pi)
    rx += dx
    ry += dy
    return out.crop((rx - crop_size[0]/2, ry - crop_size[1]/2,
                     rx + crop_size[0]/2, ry + crop_size[1]/2))

def rgb2yuv(x):
    '''convert batched rgb tensor to yuv'''
    out = x.clone()
    out[:,0,:,:] =  0.299    * x[:,0,:,:] + 0.587    * x[:,1,:,:] + 0.114   * x[:,2,:,:]
    out[:,1,:,:] = -0.168736 * x[:,0,:,:] - 0.331264 * x[:,1,:,:] + 0.5   * x[:,2,:,:]
    out[:,2,:,:] =  0.5      * x[:,0,:,:] - 0.418688 * x[:,1,:,:] - 0.081312 * x[:,2,:,:]
    return out

def yuv2rgb(x):
    '''convert batched yuv tensor to rgb'''
    out = x.clone()
    out[:,0,:,:] = x[:,0,:,:] + 1.402 * x[:,2,:,:]
    out[:,1,:,:] = x[:,0,:,:] - 0.344136 * x[:,1,:,:] - 0.714136 * x[:,2,:,:]
    out[:,2,:,:] = x[:,0,:,:] + 1.772 * x[:,1,:,:]
    return out

def yuv2ych(x):
    '''convert batched yuv tensor to ych'''
    out = x.clone()
    out[:,0,:,:] = x[:,0,:,:]
    out[:,1,:,:] = (x[:,1,:,:]**2 + x[:,2,:,:]**2)**0.5
    out[:,2,:,:] = torch.atan2(x[:,2,:,:], x[:,1,:,:])/pi/2.
    #output[:,2,:,:] += 1 * (output[:,2,:,:] < 0).type(torch.float)
    return out

def ych2yuv(x):
    '''convert batched ych tensor to yuv'''
    out = x.clone()
    h = pi*x[:,2,:,:]*2.
    out[:,0,:,:] = x[:,0,:,:]
    out[:,1,:,:] = x[:,1,:,:]*torch.cos(h)
    out[:,2,:,:] = x[:,1,:,:]*torch.sin(h)
    return out

def rgb2ych(x):
    '''convert batched rgb tensor to ych'''
    yuv = rgb2yuv(x)
    return yuv2ych(yuv)
    
def ych2rgb(x):
    '''convert batched ych tensor to rgb'''
    yuv = ych2yuv(x)
    return yuv2rgb(yuv)

def rgba2ycha(x):
    '''convert batched rgba tensor to ycha'''
    a = x[:,3:4,:,:]
    ych = rgb2ych(x[:,0:3,:,:])
    return torch.cat([ych, a], dim=1)

def ycha2rgba(x):
    '''convert batched ycha tensor to rgba'''
    a = x[:,3:4,:,:]
    rgb = ych2rgb(x[:,0:3,:,:])
    return torch.cat([rgb, a], dim=1)

def rgba2yuva(x):
    '''convert batched rgba tensor to yuva'''
    a = x[:,3:4,:,:]
    yuv = rgb2yuv(x[:,0:3,:,:])
    return torch.cat([yuv, a], dim=1)

def yuva2rgba(x):
    '''convert batched yuva tensor to rgba'''
    a = x[:,3:4,:,:]
    rgb = yuv2rgb(x[:,0:3,:,:])
    return torch.cat([rgb, a], dim=1)

def ycha2yuva(x):
    '''convert batched ycha tensor to yuva'''
    a = x[:,3:4,:,:]
    yuv = ych2yuv(x[:,0:3,:,:])
    return torch.cat([yuv, a], dim=1)

def yuva2ycha(x):
    '''convert batched yuva tensor to ycha'''
    a = x[:,3:4,:,:]
    ych = yuv2ych(x[:,0:3,:,:])
    return torch.cat([ych, a], dim=1)

colorspace_functions = {
    ("rgb", "yuv"): rgb2yuv,
    ("rgb", "ych"): rgb2ych,
    ("yuv", "ych"): yuv2ych,
    ("yuv", "rgb"): yuv2rgb,
    ("ych", "yuv"): ych2yuv,
    ("ych", "rgb"): ych2rgb,
    ("rgba", "yuva"): rgba2yuva,
    ("rgba", "ycha"): rgba2ycha,
    ("yuva", "ycha"): yuva2ycha,
    ("yuva", "rgba"): yuva2rgba,
    ("ycha", "yuva"): ycha2yuva,
    ("ycha", "rgba"): ycha2rgba,
}

def convert(x, input_colorspace, output_colorspace, clip=True):
    if clip and (output_colorspace == "rgb" or output_colorspace == "rgba"):
        return F.hardtanh(colorspace_functions[(input_colorspace.lower(), output_colorspace.lower())](x), 0, 1)
    else:
        return colorspace_functions[(input_colorspace.lower(), output_colorspace.lower())](x)

def gaussian_noise(tensor, std, mean=0.0):
    '''add gaussian noise'''
    noise = tensor.clone().normal_(mean, std)
    return tensor + noise

def scaled_gaussian_noise(tensor, std, mean=0.0, scale=2.0, mode='bilinear'):
    '''add scaled gaussian noise'''
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(0)
        unsqueezed = True
    else:
        unsqueezed = False
    noise = F.interpolate(tensor, size=(int(tensor.shape[-2]/scale), int(tensor.shape[-1]/scale)),
                          mode=mode, align_corners=False).normal_(mean, std)
    noise = F.interpolate(noise, size=(tensor.shape[-2], tensor.shape[-1]),
                          mode=mode, align_corners=False)
    out = tensor + noise
    if unsqueezed:
        out = out.squeeze(0)
    return out

def multiscale_gaussian_noise(tensor, std, mean=0.0, decay=0.9, mode='bilinear'):
    '''add multiple scaled gaussian noise (WIP experiment)'''
    scale = min(tensor.shape[-2], tensor.shape[-1])/2
    std = std
    while scale > 1:
        tensor = scaled_gaussian_noise(tensor, std, mean, scale=scale, mode=mode)
        scale /= (5**0.5 + 1)/2
        std *= decay
    return tensor

def uniform_noise(tensor, low, high):
    '''add uniform noise'''
    noise = tensor.clone().uniform_(low, high)
    return tensor + noise

def adjust_channel_brightness(tensor, channels, amount):
    '''adjust brightness of one channel'''
    out = tensor.clone()
    out[:,channels,:,:] += amount
    return out

def adjust_channel_contrast(tensor, channels, amount, rgb=True):
    '''adjust contrast of one channel'''
    out = tensor.clone()
    if rgb: out[:,channels,:,:] -= 0.5
    out[:,channels,:,:] *= amount
    if rgb: out[:,channels,:,:] += 0.5
    return out

def adjust_brightness(tensor, amount):
    '''adjust brightness of all channels'''
    out = tensor.clone()
    out += amount
    return out

def adjust_contrast(tensor, amount):
    '''adjust contrast of all channels'''
    out = tensor.clone()
    out -= 0.5
    out *= amount
    out += 0.5
    return out
