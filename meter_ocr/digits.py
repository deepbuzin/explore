import os
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from mat73 import loadmat
import h5py
from torchvision.datasets import CocoDetection
# from torchvision import transforms as T
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from engine import train_one_epoch, evaluate
import utils


class SvhnDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.anno = pd.read_csv('d.csv')

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        anno = self.anno.loc[self.anno['name'] == self.imgs[idx]]

        # targets = []

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for _, a in anno.iterrows():
            xmin = a['left']
            ymin = a['top']
            xmax = a['left'] + a['width']
            ymax = a['top'] + a['height']

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(a['label'])
            area.append(a['width'] * a['height'])
            iscrowd.append(0)


            # targets.append({
            #     'image_id': torch.tensor([idx]),
            #     'boxes': torch.as_tensor([xmin, ymin, xmax, ymax], dtype=torch.float32),
            #     'labels': torch.as_tensor(a['label'], dtype=torch.int64),
            #     'area': torch.as_tensor(a['width'] * a['height'], dtype=torch.float32),
            #     'iscrowd': torch.as_tensor([0], dtype=torch.int64),
            # })

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([idx])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


def get_dataset():
    dataset = SvhnDataset(root='C:\\Users\\Andrey\\Desktop\\svhn',
                          transforms=get_transform(train=True))

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:10000])

    print('Number of samples: ', len(dataset))
    img, target = dataset[111]
    return dataset


def get_transform(train):
    return T.Compose([
            T.Rescale(128),
            T.RandomCrop((128, 128)),
            T.ToTensor()
        ])


def get_model():
    num_classes = 11
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = get_dataset()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

    model = get_model()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=1)
        lr_scheduler.step()
        torch.save(model.state_dict(), f'digits_model_{epoch}.pth')


if __name__ == "__main__":
    main()