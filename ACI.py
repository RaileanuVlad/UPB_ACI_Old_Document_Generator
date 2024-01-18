import pandas as pd
import numpy as np

from glob import glob
from skimage.util import random_noise
import skimage.exposure
from imgaug import augmenters as iaa

import cv2
import matplotlib.pylab as plt
import random
from numpy.random import default_rng
from scipy import ndimage

imageIndex = 2
rows = 4
columns = 3
currentImg = 1

kernelX = random.randint(2,4)
kernelY = random.randint(2,4)
kernel = np.ones((kernelX,kernelY), np.uint8)
blurX = random.randint(5,7)
blurY = random.randint(5,7)

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

if __name__ == "__main__":
    
    docs = glob('.\docs\*.jpg')
    dust = glob('.\dust\*.png')
    dustImg = cv2.imread(dust[random.randint(0,len(dust)-1)], cv2.IMREAD_UNCHANGED)
    
    fig = plt.figure(figsize=(10,10))
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    image = cv2.imread(docs[imageIndex])
    plt.title("image")
    plt.imshow(image)
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    colorImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title("colorImage")
    plt.imshow(colorImage)
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_RGB2GRAY)
    plt.title("grayImage")
    plt.imshow(grayImage, cmap='gray')
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    bigImage = cv2.resize(grayImage, None, fx = 2, fy = 2, interpolation=cv2.INTER_LANCZOS4)
    plt.title("bigImage")
    plt.imshow(bigImage, cmap='gray')
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    dilatedImage = cv2.dilate(bigImage,kernel)
    plt.title("dilatedImage")
    plt.imshow(dilatedImage, cmap='gray')
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    borderedImage = cv2.copyMakeBorder(dilatedImage, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=0)
    plt.title("borderedImage")
    plt.imshow(borderedImage, cmap='gray')
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    # h, w, d = noiseImage.shape
    # dustImg = cv2.resize(dustImg, (w, h)).astype("float64")
    dustedImage = cv2.cvtColor(borderedImage, cv2.COLOR_GRAY2RGB)
    h, w, d = dustImg.shape
    add_transparent_image(dustedImage, dustImg, random.randint(0,w/4), -random.randint(0,h))
    plt.title("dustedImage")
    plt.imshow(dustedImage)
    currentImg += 1
    
    
    
    
    
    
    
    height, width = dustedImage.shape[:2]

    seedval = random.randint(0, 100000)
    rng = default_rng(seed=seedval)

    noise = rng.integers(0, 255, (height,width), np.uint8, True)

    blur = cv2.GaussianBlur(noise, (0,0), sigmaX=15, sigmaY=15, borderType = cv2.BORDER_DEFAULT)

    stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0,190)).astype(np.uint8)

    thresh = cv2.threshold(stretch, 141, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.merge([mask,mask,mask])

    result1 = cv2.add(dustedImage, mask)

    edges = cv2.Canny(mask,50,255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, kernel)
    edges = cv2.merge([edges,edges,edges])

    result2 = result1.copy()
    result2[np.where((edges == [255,255,255]).all(axis=2))] = [0,0,0]

    noise = cv2.merge([noise,noise,noise])
    stainedImage = result2.copy()
    stainedImage = np.where(mask==(255,255,255), noise, stainedImage)
    
    
    
    
    
    

    
    
    
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    blurredImage = cv2.blur(stainedImage, [blurX, blurY], 0)
    plt.title("blurredImage")
    plt.imshow(blurredImage, cmap='gray')
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    tint_color = np.array([0, random.randint(0,130), 255], dtype=np.uint8)
    tint_factor = random.randint(250,500)/1000
    tintedImage = blurredImage
    tinted_mask  = np.full(tintedImage.shape, tint_color, np.uint8)
    # tintedImage = cv2.addWeighted(tintedImage, 1 - tint_factor, tinted_mask, tint_factor, 0)
    tintedImage = cv2.subtract(tintedImage, tinted_mask)
    tintedImageHSV = cv2.cvtColor(tintedImage, cv2.COLOR_RGB2HSV).astype("float32")
    (h, s, v) = cv2.split(tintedImageHSV)
    s = s*tint_factor
    s = np.clip(s,0,255)
    tintedImageHSV = cv2.merge([h,s,v])
    tintedImage = cv2.cvtColor(tintedImageHSV.astype("uint8"), cv2.COLOR_HSV2RGB)
    # tintedImage = cv2.cvtColor(tintedImage.astype("uint8"), cv2.COLOR_RGB2RGBA)
    plt.title("tintedImage")
    plt.imshow(tintedImage)
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    noiseImage = random_noise(tintedImage, mode="gaussian", clip=True)
    plt.title("noiseImage")
    plt.imshow(noiseImage)
    currentImg += 1
    
    fig.add_subplot(rows, columns, currentImg)
    plt.axis('off')
    rotatedImage = ndimage.rotate(noiseImage, random.randint(-10, 10), mode="constant", cval=255.0)
    plt.title("rotatedImage")
    plt.imshow(rotatedImage)
    currentImg += 1
    
    plt.show()

    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(rotatedImage)
    ax.axis('off')
    
    plt.show()
    
    cv2.imwrite('./results/dust.png', dustImg)
    cv2.imwrite('./results/result.png', cv2.cvtColor(rotatedImage.astype('int'), cv2.COLOR_RGB2BGR))