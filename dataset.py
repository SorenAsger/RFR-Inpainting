import os
import glob
import scipy
import torch
import random
import numpy as np
import torchvision.transforms.functional as F
from PIL import Image
# from scipy.misc import imread
from imageio import imread
import cv2


class Dataset(torch.utils.data.Dataset):
    def __init__(self, image_path, mask_path, mask_mode, target_size, augment=True, training=True, mask_reverse=False):
        super(Dataset, self).__init__()
        self.augment = augment
        self.training = training
        self.data = self.load_list(image_path)
        self.mask_data = self.load_list(mask_path)

        self.target_size = target_size
        self.mask_type = mask_mode
        self.mask_reverse = mask_reverse
        if not training:
            random.seed(69696969)
            np.random.seed(69696969)
        # in test mode, there's a one-to-one relationship between mask and image
        # masks are loaded non random

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        try:
            item = self.load_item(index)
        except:
            print('loading error: ' + self.data[index])
            item = self.load_item(0)

        return item

    def load_item(self, index):
        # print("I am here 1")
        img = imread(self.data[index])
        if self.training:
            img = self.resize(img)
        else:
            img = self.resize(img, True, True, True)
        # load mask
        mask = self.load_mask(img, index)
        # print("I am here 2")

        # augment data
        if self.training:
            if self.augment and np.random.binomial(1, 0.5) > 0:
                img = img[:, ::-1, ...]
            if self.augment and np.random.binomial(1, 0.5) > 0:
                mask = mask[:, ::-1, ...]
        # print("I am here 3")

        return self.to_tensor(img), self.to_tensor(mask)

    def load_mask(self, img, index):
        imgh, imgw = img.shape[0:2]

        # external mask, random order
        if self.mask_type == 0:
            mask_index = random.randint(0, len(self.mask_data) - 1)
            mask = imread(self.mask_data[mask_index])
            mask = (mask > 0).astype(np.uint8)  # threshold due to interpolation
            mask = self.resize(mask, False)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255
        # generate random mask
        if self.mask_type == 1:
            mask = 1 - generate_stroke_mask([self.target_size, self.target_size])
            mask = (mask > 0).astype(np.uint8) * 255
            mask = self.resize(mask, False)
            return mask

        # external mask, fixed order
        if self.mask_type == 2:
            mask_index = index
            mask = imread(self.mask_data[mask_index])
            mask = (mask > 0).astype(np.uint8)  # threshold due to interpolation
            mask = self.resize(mask, False)
            if self.mask_reverse:
                return (1 - mask) * 255
            else:
                return mask * 255

        # Our generate mask, squares + lines, this is 10-20 %
        if self.mask_type == 3:
            m = gen_random_square_lines_mask(self.target_size, 1)
            m = self.resize(m, False)
            return m

        if self.mask_type == 4:  # this is 50-60%
            m = gen_random_square_lines_mask(self.target_size, 2)
            m = self.resize(m, False)
            return m

    def resize(self, img, aspect_ratio_kept=True, fixed_size=False, centerCrop=False):
        if aspect_ratio_kept:
            imgh, imgw = img.shape[0:2]
            side = np.minimum(imgh, imgw)
            if fixed_size:
                if centerCrop:
                    # center crop
                    j = (imgh - side) // 2
                    i = (imgw - side) // 2
                    img = img[j:j + side, i:i + side, ...]
                else:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
            else:
                if side <= self.target_size:
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = 0
                    w_start = 0
                    if j != 0:
                        h_start = random.randrange(0, j)
                    if i != 0:
                        w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
                else:
                    side = random.randrange(self.target_size, side)
                    j = (imgh - side)
                    i = (imgw - side)
                    h_start = random.randrange(0, j)
                    w_start = random.randrange(0, i)
                    img = img[h_start:h_start + side, w_start:w_start + side, ...]
        # img = scipy.misc.imresize(img, [self.target_size, self.target_size])
        img = np.array(Image.fromarray(img).resize(size=(self.target_size, self.target_size)))
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            print("Found something with alpha channel")
            img = img[:, :, :3]
        return img

    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def load_list(self, path):
        if isinstance(path, str):
            if path[-3:] == "txt":
                line = open(path, "r")
                lines = line.readlines()
                file_names = []
                for line in lines:
                    file_names.append("../../Dataset/Places2/train/data_256" + line.split(" ")[0])
                return file_names
            if os.path.isdir(path):
                path = list(glob.glob(path + '/*.jpg')) + list(glob.glob(path + '/*.png')) + list(
                    glob.glob(path + '/*.JPEG')) + list(glob.glob(path + '/*.PNG'))
                path.sort()
                return path
            if os.path.isfile(path):
                try:
                    return np.genfromtxt(path, dtype=np.str, encoding='utf-8')
                except:
                    return [path]
        return []


def generate_stroke_mask(im_size, max_parts=15, maxVertex=25, maxLength=100, maxBrushWidth=24, maxAngle=360):
    mask = np.zeros((im_size[0], im_size[1], 1), dtype=np.float32)
    parts = random.randint(1, max_parts)
    for i in range(parts):
        mask = mask + np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, im_size[0], im_size[1])
    mask = np.minimum(mask, 1.0)
    mask = np.concatenate([mask, mask, mask], axis=2)
    return mask


def gen_random_square_lines_mask(size, cover):
    def gen():
        width = size
        height = width
        if cover == 1:  # 10-20 %
            h_s, w_s = random.randrange(height // 2), random.randrange(width // 2)
            h_d, w_d = random.randint(height // 8, height // 4), random.randint(width // 8, width // 4)
            h_e, w_e = h_s + h_d, w_s + w_d
            lower_thick_limit = 0.02  # for lines
            upper_thick_limit = 0.05
        else:  # 50-60%
            h_s, w_s = random.randrange(height // 2), random.randrange(width // 2)
            h_d, w_d = random.randint(height // 3, height // 1.5), random.randint(width // 3, width // 1.5)
            h_e, w_e = h_s + h_d, w_s + w_d
            lower_thick_limit = 0.05
            upper_thick_limit = 0.12
        m = np.ones((height, width, 1), dtype=np.float32)
        m[h_s:h_e, w_s:w_e] = 0
        for _ in range(np.random.randint(8, 16)):
            # Get random x locations to start line

            x1, x2 = np.random.randint(1, width), np.random.randint(1, width)

            # Get random y locations to start line

            y1, y2 = np.random.randint(1, height), np.random.randint(1, height)

            # Get random thickness of the line drawn

            thickness = np.random.randint(width * lower_thick_limit, width * upper_thick_limit)

            # Draw black line on the white mask

            cv2.line(m, (x1, y1), (x2, y2), (0), thickness)
        m = np.concatenate([m, m, m], axis=2)
        m = (m > 0).astype(np.uint8) * 255
        return m

    while True:
        m = gen()
        mcov = mask_cover(m)
        if cover == 1 and (0.1 < mcov < 0.2):
            break
        if cover != 1 and (0.5 < mcov < 0.6):
            break
    return m


def np_free_form_mask(maxVertex, maxLength, maxBrushWidth, maxAngle, h, w):
    mask = np.zeros((h, w, 1), np.float32)
    numVertex = np.random.randint(maxVertex + 1)
    startY = np.random.randint(h)
    startX = np.random.randint(w)
    brushWidth = 0
    for i in range(numVertex):
        angle = np.random.randint(maxAngle + 1)
        angle = angle / 360.0 * 2 * np.pi
        if i % 2 == 0:
            angle = 2 * np.pi - angle
        length = np.random.randint(maxLength + 1)
        brushWidth = np.random.randint(10, maxBrushWidth + 1) // 2 * 2
        nextY = startY + length * np.cos(angle)
        nextX = startX + length * np.sin(angle)
        nextY = np.maximum(np.minimum(nextY, h - 1), 0).astype(np.int)
        nextX = np.maximum(np.minimum(nextX, w - 1), 0).astype(np.int)
        cv2.line(mask, (startY, startX), (nextY, nextX), 1, brushWidth)
        cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
        startY, startX = nextY, nextX
    cv2.circle(mask, (startY, startX), brushWidth // 2, 2)
    return mask


def mask_cover(mask):
    return np.mean((mask == 0))


# gen_random_square_lines_mask(256, 2)
"""
msk = gen_random_square_lines_mask(256, 50)
print((msk > 0).sum())
print((msk > 0).shape)
print(256*256*3)
#smsk = generate_stroke_mask([256, 256])
#print(smsk)
mask = gen_random_square_lines_mask(256, 50)
mask = (mask > 0).astype(np.uint8) * 255
print(mask)
print(mask_cover(mask))
"""
