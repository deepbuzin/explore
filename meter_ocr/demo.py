import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

model.load_state_dict(torch.load('model_9.pth'))
model.to('cuda:0')
model.eval()

orig_img = cv2.imread('data_2\\id_82_value_1100_014.jpg')
# orig_img = cv2.imread('data\\cold_water\\cold_80.png')
h, w, _ = orig_img.shape
if h > w:
    new_h, new_w = 512 * h / w, 512
else:
    new_h, new_w = 512, 512 * w / h

new_h, new_w = int(new_h), int(new_w)
orig_img = cv2.resize(orig_img, (new_w, new_h))

img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
img = img / 255
img = img - np.array([0.485, 0.456, 0.406])
img = img / np.array([0.229, 0.224, 0.225])

_img = torch.as_tensor(img)
_img = _img.permute(2, 0, 1)

# _img = F.normalize(_img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
_img = torch.unsqueeze(_img, 0).float().to('cuda:0')

# x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(_img)

for p in predictions:
    boxes = p['boxes'].detach().cpu().numpy()
    labels = p['labels'].detach().cpu().numpy()
    for b, l in zip(boxes, labels):
        if l == 1:
            color = (255, 144, 0)
        elif l == 2:
            color = (0, 144, 255)
        else:
            color = (144, 0, 255)
        cv2.rectangle(orig_img, (b[0], b[1]), (b[2], b[3]), color, 2)

cv2.imshow('frame', orig_img)
cv2.waitKey(0)

print(predictions)
