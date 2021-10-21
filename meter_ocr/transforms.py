import random
import torch
import numpy as np
from PIL import Image

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image, landmarks):
        w, h = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_w, new_h))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        if type(landmarks) is list:
            for l in landmarks:
                l['boxes'] = l['boxes'] * torch.as_tensor([new_w / w, new_h / h, new_w / w, new_h / h])
        else:
            landmarks['boxes'] = landmarks['boxes'] * torch.as_tensor([new_w / w, new_h / h, new_w / w, new_h / h])

        return img, landmarks


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image, landmarks):
        w, h = image.size
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h) if new_h < h else 0
        left = np.random.randint(0, w - new_w) if new_w < w else 0

        image = image.crop((left, top, left + new_w, top + new_h))

        if type(landmarks) is list:
            for l in landmarks:
                l['boxes'] = l['boxes'] - torch.as_tensor([left, top, left, top])
        else:
            landmarks['boxes'] = landmarks['boxes'] - torch.as_tensor([left, top, left, top])

        # landmarks = landmarks - [left, top]

        return image, landmarks


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
