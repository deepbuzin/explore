import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.datasets import CocoDetection
# from torchvision import transforms as T
import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from engine import train_one_epoch, evaluate
import utils


class MeterDataset(CocoDetection):
    # def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
    #     super(MeterDataset, self).__init__(root, transforms, transform, target_transform)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        area = []
        iscrowd = []

        for t in target:
            bbox = [t['bbox'][0],
                    t['bbox'][1],
                    t['bbox'][0] + t['bbox'][2],
                    t['bbox'][1] + t['bbox'][3]]

            boxes.append(bbox)
            labels.append(t['category_id'])
            area.append(t['area'])
            iscrowd.append(t['iscrowd'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        image_id = torch.tensor([target[0]['image_id']])

        _target = {}
        _target["boxes"] = boxes
        _target["labels"] = labels
        _target["image_id"] = image_id
        _target["area"] = area
        _target["iscrowd"] = iscrowd

            # item = {'boxes': bbox,
            #         'labels': torch.as_tensor(t['category_id'], dtype=torch.int64),
            #         'image_id': torch.as_tensor(t['image_id'], dtype=torch.int64),
            #         'iscrowd': torch.as_tensor(t['iscrowd'], dtype=torch.uint8),
            #         'area': torch.as_tensor(t['area'], dtype=torch.float)}
            # _target.append(item)


        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        # if self.transforms is not None:
        img, _target = self.transforms(img, _target)
        img = F.normalize(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return img, _target


def get_dataset():
    dataset = MeterDataset(root='C:\\Users\\Andrey\\Desktop\\meter_data\\meter\\meter1',
                           annFile='C:\\Users\\Andrey\\Desktop\\meter_data\\meter\\annotations\\instances_meter1.json',
                           transforms=get_transform(train=True))


    print('Number of samples: ', len(dataset))
    # img, target = dataset[111]
    return dataset


def get_transform(train):
    return T.Compose([
            T.Rescale(512),
            # T.RandomCrop((512, 512)),
            T.ToTensor()
        ])


def get_model():
    num_classes = 4
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset = get_dataset()

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=utils.collate_fn)

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
        torch.save(model.state_dict(), f'model_{epoch}.pth')





if __name__ == "__main__":
    main()