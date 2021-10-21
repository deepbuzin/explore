import torch
import numpy as np
from PIL import Image
import os
import glob
import tqdm
from efficientnet_pytorch import EfficientNet, get_model_params

from beards_trainer import BeardsTrainer

BEARDS_CLASS_NAMES = ['chin_curtain', 'classic_long', 'classic_short', 'goatee', 'goatee_with_moustache', 'moustache',
                      'shaven', 'stubble']


class Beards:
    def __init__(self, model_name, class_names, state_dict=None):
        self.model_name = model_name
        self.class_names = class_names

        self._model = EfficientNet.from_pretrained(self.model_name, num_classes=len(self.class_names))
        if state_dict is not None:
            self._model.load_state_dict(torch.load(state_dict))
        self._model.eval()

        _, global_params = get_model_params(self.model_name, override_params=None)
        self._image_size = global_params.image_size
        self._transforms = BeardsTrainer.get_transforms(self._image_size)

    def infer(self, image):
        with torch.no_grad():
            image_infer = self._transforms(image).unsqueeze(0)
            outputs = self._model(image_infer)
            probs = torch.softmax(outputs, dim=1)
            pred_id = np.argmax(np.squeeze(probs.numpy()))
            pred = self.class_names[int(pred_id)]
        return pred

    def val_errors(self, data_dir):
        paths = glob.glob(os.path.join(data_dir, 'val', '*', '*.png'))
        error_counter = 0
        with torch.no_grad():
            for idx, img_path in tqdm.tqdm(enumerate(paths), total=len(paths)):
                img_orig = Image.open(img_path)
                img = self._transforms(img_orig).unsqueeze(0)

                gt = os.path.basename(os.path.dirname(img_path))

                outputs = self._model(img)
                probs = torch.softmax(outputs, dim=1)
                pred_id = np.argmax(np.squeeze(probs.numpy()))
                pred = self.class_names[int(pred_id)]

                if pred != gt:
                    error_counter += 1
                    if not os.path.exists(os.path.join('errors', gt)):
                        os.makedirs(os.path.join('errors', gt))
                    img_orig.save(os.path.join('errors', gt, '{}_{}.png'.format(pred, idx)))
                    print()
                    print('Wrong class for {}'.format(img_path))
                    for i, p in enumerate(np.squeeze(probs.numpy())):
                        print('{:>22}: {:1.2f}'.format(self.class_names[i], p))
        print('Acc: {:.2f}'.format(1 - (error_counter / len(paths))))


if __name__ == '__main__':
    b = Beards(model_name='efficientnet-b5',
               class_names=BEARDS_CLASS_NAMES,
               state_dict='b5/model_best.pth')
    b.val_errors('data')
