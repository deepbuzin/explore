import torch
import torchvision
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_meter_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

    model.load_state_dict(torch.load('model_9.pth'))
    model.to('cuda:0')
    model.eval()
    return model


def get_digit_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 11)

    model.load_state_dict(torch.load('digits_model_0.pth'))
    model.to('cuda:0')
    model.eval()
    return model


class Processor:
    def __init__(self):
        self.meter_model = get_meter_model()
        self.digit_model = get_digit_model()

    @staticmethod
    def _preprocess_image(img):
        h, w, _ = img.shape
        if h > w:
            new_h, new_w = 512 * h / w, 512
        else:
            new_h, new_w = 512, 512 * w / h

        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(img, (new_w, new_h))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        img = img - np.array([0.485, 0.456, 0.406])
        img = img / np.array([0.229, 0.224, 0.225])

        _img = torch.as_tensor(img)
        _img = _img.permute(2, 0, 1)

        _img = torch.unsqueeze(_img, 0).float().to('cuda:0')
        return _img, new_w / w

    @staticmethod
    def _postprocess_meter(predictions):
        results = list()

        for p in predictions:
            boxes = p['boxes'].detach().cpu().numpy()
            labels = p['labels'].detach().cpu().numpy()
            scores = p['scores'].detach().cpu().numpy()
            for b, l, s in zip(boxes, labels, scores):
                results.append((b, l, s))
        return results

    @staticmethod
    def _postprocess_digits(predictions):
        results = list()

        for p in predictions:
            boxes = p['boxes'].detach().cpu().numpy()
            labels = p['labels'].detach().cpu().numpy()
            scores = p['scores'].detach().cpu().numpy()
            for b, l, s in zip(boxes, labels, scores):
                results.append((b, l, s))
        return results

    def process(self, img_path):
        img = cv2.imread(img_path)
        _img, scale = self._preprocess_image(img.copy())

        predictions = self.meter_model(_img)
        meter_results = self._postprocess_meter(predictions)

        for r in meter_results:
            bbox, label, score = r
            if score <= 0.4:
                continue
            digit_img = _img[:, bbox[2]:bbox[3], bbox[0]:bbox[1]].detach().clone()
            digit_preds = self.digit_model(digit_img)






