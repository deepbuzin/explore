import numpy as np
from tqdm import tqdm
from glob import glob
import os
import json

LABELS_PATH = 'data/labels_for_test.json'
IMAGES_PATH = 'data/data_for_test'
SPLIT = 0.1

np.random.seed(777)
image_paths = glob(os.path.join(IMAGES_PATH, '*.png'))

with open(LABELS_PATH) as f:
    labels = json.load(f)
    for image_path in tqdm(image_paths, total=len(image_paths)):
        filename = os.path.basename(image_path)
        label = labels[filename]
        split = 'val' if (np.random.rand()) < SPLIT else 'train'

        if not os.path.exists(os.path.join('data', split, label)):
            os.makedirs(os.path.join('data', split, label))

        os.rename(image_path, os.path.join('data', split, label, filename))


