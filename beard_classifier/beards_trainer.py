import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torchvision import datasets, transforms
import time
import os
import copy

from efficientnet_pytorch import EfficientNet, get_model_params

np.random.seed(777)

IMAGENET_PARAMS = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class BeardsTrainer:
    def __init__(self, model_name='efficientnet-b5'):
        self.model_name = model_name
        self._model = None
        self._image_size = None
        self._device = None

        self._transforms = None
        self._datasets = None
        self._loaders = None
        self._dataset_sizes = None
        self._num_classes = None
        self._class_names = None

    @staticmethod
    def get_transforms(image_size, for_train=False):
        if for_train:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_PARAMS)
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(**IMAGENET_PARAMS)
            ])

    def train(self, data_dir='data',
              batch_size=3,
              num_epochs=50,
              from_checkpoint=None,
              weight=None,
              optim_params=dict(lr=1e-3, momentum=0.9),
              lr_scheduler_params=dict(step_size=1, gamma=0.9)):
        self._init(data_dir, batch_size)
        if from_checkpoint is not None:
            print('Loading {}'.format(from_checkpoint))
            self._model.load_state_dict(torch.load(from_checkpoint))
        model = self._model.to(self._device)

        if weight is None:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.asarray(weight)).float().to(self._device))

        optimizer = optim.SGD(model.parameters(), **optim_params)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, **lr_scheduler_params)

        if not os.path.exists('training'):
            os.makedirs('training')
        self._do_train(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

    def _init(self, data_dir, batch_size):
        _, global_params = get_model_params(self.model_name, override_params=None)
        self._image_size = global_params.image_size
        self._transforms = {'train': self.get_transforms(self._image_size, for_train=True),
                            'val': self.get_transforms(self._image_size)}

        self._datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  self._transforms[x])
                          for x in ['train', 'val']}

        self._loaders = {x: torch.utils.data.DataLoader(self._datasets[x], batch_size=batch_size,
                                                        shuffle=True, num_workers=4)
                         for x in ['train', 'val']}
        self._dataset_sizes = {x: len(self._datasets[x]) for x in ['train', 'val']}
        self._class_names = self._datasets['train'].classes
        self._num_classes = len(self._class_names)

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model = EfficientNet.from_pretrained(self.model_name, num_classes=self._num_classes)

    def _do_train(self, model, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for idx, (inputs, labels) in enumerate(self._loaders[phase]):
                    inputs = inputs.to(self._device)
                    labels = labels.to(self._device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    print('Epoch {} :: {}/{} :: Loss: {:.4f}'
                          .format(epoch, idx, self._dataset_sizes[phase], running_loss / (idx + 1)))

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / self._dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self._dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'training/model_best.pth')
                torch.save(model.state_dict(), 'training/model_{}.pth'.format(epoch))

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        model.load_state_dict(best_model_wts)
        return model


if __name__ == '__main__':
    trainer = BeardsTrainer(model_name='efficientnet-b5')
    trainer.train(data_dir='data',
                  batch_size=3,
                  num_epochs=50,
                  from_checkpoint='b5/model_best.pth',
                  weight=[1., 1., 1.15, 1., 1., 1.1, 1., 1.1])
