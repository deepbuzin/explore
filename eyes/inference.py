import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import Lambda

import numpy as np
import pickle
from pathlib import Path
import argparse


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


class EyeDataset(Dataset):
    def __init__(self, img_dir: Path, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.img_list = list(img_dir.glob('*.jpg'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        img = read_image(img_path.as_posix())
        if self.transform:
            img = self.transform(img)
        return img


class EyeEncoder(nn.Module):
    def __init__(self):
        super(EyeEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, (7, 7), padding='same'),  # 128 x 24 x 24
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2)),  # 128 x 12 x 12
            nn.Conv2d(128, 32, (3, 3), padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2, 2)),  # 32 x 6 x 6
            nn.Conv2d(32, 1, (7, 7), padding='same'),  # 1 x 6 x 6
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


def load(param_dir: Path):
    encoder = EyeEncoder()
    encoder.load_state_dict(torch.load(param_dir.joinpath(Path('encoder.pth'))))
    encoder.to(device)

    with param_dir.joinpath(Path('clustering.pkl')).open('rb') as f:
        clustering = pickle.load(f)

    return encoder, clustering


def init_data(img_dir: Path):
    def div_by_255(x):
        return x / 255.

    dataset = EyeDataset(img_dir, transform=Lambda(lambda x: div_by_255(x)))
    dataloader = DataLoader(dataset, batch_size=128)

    return dataset, dataloader


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Infer eyes')
    parser.add_argument('img_dir', type=str, help='dir with images to infer')
    parser.add_argument('--param_dir', type=str, help='dir with encoder weights and clustering model', default='params')
    parser.add_argument('--invert_labels', action='store_true', help='set open to 0 and closed to 1')
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    param_dir = Path(args.param_dir)

    encoder, clustering = load(param_dir)
    dataset, dataloader = init_data(img_dir)

    embs = []

    for x in dataloader:
        x = x.to(device)
        x_emb = encoder(x)
        x_emb = torch.flatten(x_emb, start_dim=1)
        embs.append(x_emb.detach().cpu().numpy())
    embs = np.concatenate(embs, axis=0)

    labels = clustering.predict(embs)
    if args.invert_labels:
        labels = 1 - labels

    out_path = Path('result.csv')

    with out_path.open('w') as f:
        for label, img_path in zip(labels, dataset.img_list):
            f.write(f'{img_path},{label}\n')


