import random
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms.functional as F
from dataset.autoaug import AutoAugment
import numbers
import torchvision

class MultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img):
        im_size = img.size
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h))
        ret_img_group = crop_img_group.resize((self.input_size[0], self.input_size[1]), self.interpolation)
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    def __str__(self):
        return self.__class__.__name__
class Pad_(object):
    def __init__(self):
        self.w = 192
        self.h = 256

    def __call__(self, image):

        w_1, h_1 = image.size
        #print( w_1, h_1)
        ratio_f = self.w / self.h
        ratio_1 = w_1 / h_1

        #image = T.ToTensor()(image)
        # check if the original and final aspect ratios are the same within a margin
        if round(ratio_1, 2) != round(ratio_f, 2):

            # padding to preserve aspect ratio
            hp = int(w_1/ratio_f - h_1)
            wp = int(ratio_f * h_1 - w_1)
            if hp > 0 and wp < 0:
                hp = hp // 2
                #image = F.pad(image, (0, hp, 0, hp) ,0, 'constant')
                #print(image)
                #return image
                #print(T.ToPILImage()(image))
                #return T.ToPILImage()(image)
                #return F.resize(image, [self.h, self.w])
                #return (int(0), int(hp), int(0), int(hp))
                return transforms.functional.pad(image, (int(0), int(hp), int(0), int(hp)) ,0, 'constant')

            elif hp < 0 and wp > 0:
                #wp = wp // 2
                #image = F.pad(image, (wp, 0, wp, 0), 0, 'constant')
                #return T.ToPILImage()(image)
                #return F.resize(image, [self.h, self.w])
                #print(image)
                #return image
                #(int(wp), int(0), int(wp), int(0))
                #return (int(wp), int(0), int(wp), int(0))
                return F.pad(image, (int(wp), int(0), int(wp), int(0)), 0, 'constant')
        # else:
        #     #print(image)
        #     return (int(0), int(0), int(0), int(0))

def get_padding(image):    
    w_1, h_1 = image.size
    #print( w_1, h_1)
    ratio_f = 192 / 256
    ratio_1 = w_1 / h_1

    #image = T.ToTensor()(image)
    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):

        # padding to preserve aspect ratio
        hp = int(w_1/ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        #print(hp,wp)
        if hp > 0 and wp <= 0:
            hp = hp // 2
            #image = F.pad(image, (0, hp, 0, hp) ,0, 'constant')
            #print(image)
            #return image
            #print(T.ToPILImage()(image))
            #return T.ToPILImage()(image)
            #return F.resize(image, [self.h, self.w])
            #return (int(0), int(hp), int(0), int(hp))
            return (int(0), int(hp), int(0), int(hp))

        elif hp <= 0 and wp > 0:
            wp = wp // 2
            #image = F.pad(image, (wp, 0, wp, 0), 0, 'constant')
            #return T.ToPILImage()(image)
            #return F.resize(image, [self.h, self.w])
            #print(image)
            #return image
            #(int(wp), int(0), int(wp), int(0))
            #return (int(wp), int(0), int(wp), int(0))
            return (int(wp), int(0), int(wp), int(0))
        else:
            return (int(0), int(0), int(0), int(0))
    else:
        return (int(0), int(0), int(0), int(0))
    

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple,list))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.w = 192
        self.h = 256
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        #print(type(get_padding(img)))
        #print(img)
        #print(self.fill)
        return torchvision.transforms.functional.pad(img, get_padding(img), self.fill, self.padding_mode)
    
    # def __repr__(self):
    #     return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
    #         format(self.fill, self.padding_mode) 
def get_transform(cfg):
    height = cfg.DATASET.HEIGHT
    width = cfg.DATASET.WIDTH
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if cfg.DATASET.TYPE == 'pedes':

        train_transform = T.Compose([
            #T.Resize((height, width)),
            #T.Pad(10),
            #T.RandomGrayscale(),
            #Pad(w=width, h=height),
            #T.Pad([Pad_(w=width, h=height)]),
            NewPad(),
            T.Resize((height, width)),
            T.RandomGrayscale(),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
            T.RandomErasing(),
        ])
 
        valid_transform = T.Compose([
            #T.Resize((height, width)),
            #T.Grayscale(num_output_channels = 3 ),
            #Pad(w=width, h=height),
            NewPad(),
            T.Resize((height, width)),
            #T.Pad(10),
            #T.Grayscale(num_output_channels = 3 ),
            T.RandomGrayscale(),
            T.ToTensor(),
            normalize
        ])
        valid_transform1 = T.Compose([
            #T.Resize((height, width)),
            #T.Grayscale(num_output_channels = 3 ),
            #Pad(w=width, h=height),
            NewPad(),
            T.Resize((height, width)),
            #T.Pad(10),
            T.Grayscale(num_output_channels = 3 ),
            #T.RandomGrayscale(),
            T.ToTensor(),
            normalize
        ])

    elif cfg.DATASET.TYPE == 'multi_label':

        valid_transform = T.Compose([
            T.Resize([height, width]),
            T.ToTensor(),
            normalize,
        ])

        if cfg.TRAIN.DATAAUG.TYPE == 'autoaug':
            train_transform = T.Compose([
                T.RandomApply([AutoAugment()], p=cfg.TRAIN.DATAAUG.AUTOAUG_PROB),
                T.Resize((height, width), interpolation=3),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])

        else:
            train_transform = T.Compose([
                T.Resize((height + 64, width + 64)),
                MultiScaleCrop(height, scales=(1.0, 0.875, 0.75, 0.66, 0.5), max_distort=2),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
    else:

        assert False, 'xxxxxx'

    return train_transform, valid_transform #,valid_transform1
