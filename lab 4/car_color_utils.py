import os
import re
import numpy as np
from collections import Counter
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.models as models


def extract_color_from_filename(filename):
    name = filename.replace('.png', '').replace('.jpg', '')
    parts = re.split(r'[_-]', name.lower())

    color_keywords = {
        'black': ['black', 'blk', 'noir', 'schwarz'],
        'white': ['white', 'wht', 'blanc', 'weiss', 'pearl'],
        'silver': ['silver', 'slv', 'argent', 'silber', 'gray', 'grey'],
        'red': ['red', 'rouge', 'rot', 'rosso', 'maroon'],
        'blue': ['blue', 'blu', 'bleu', 'blau', 'azul'],
        'green': ['green', 'grn', 'vert', 'grun'],
        'yellow': ['yellow', 'ylw', 'jaune', 'gelb'],
        'brown': ['brown', 'brn', 'brun', 'braun', 'beige', 'tan'],
        'gold': ['gold', 'gld', 'or'],
    }

    for part in parts:
        for color, keywords in color_keywords.items():
            if any(keyword in part for keyword in keywords):
                return color
            
    return None

def load_dvm_dataset(data_dir, min_samples_per_class=500):
    print("Loading DVM dataset...")
    image_paths, colors, metadata = [], [], []

    brands = [b for b in os.listdir(data_dir)
              if os.path.isdir(os.path.join(data_dir, b))]

    for brand in tqdm(brands):
        brand_path = os.path.join(data_dir, brand)
        years = [y for y in os.listdir(brand_path)
                 if os.path.isdir(os.path.join(brand_path, y))]
        
        for year in years:
            year_path = os.path.join(brand_path, year)
            images = [f for f in os.listdir(year_path)
                      if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in images:
                color = extract_color_from_filename(img_file)

                if color is not None:
                    image_paths.append(os.path.join(year_path, img_file))
                    colors.append(color)
                    metadata.append({'brand': brand, 'year': year,
                                     'filename': img_file})

    print(f"\nImages found with color tags: {len(image_paths)}")
    color_counts = Counter(colors)
    print("\nColor distribution:")

    for color, count in sorted(color_counts.items(),
                                key=lambda x: x[1], reverse=True):
        print(f"  {color}: {count}")

    valid_colors = {c for c, n in color_counts.items()
                    if n >= min_samples_per_class}
    
    if len(valid_colors) < len(color_counts):
        print(f"\nExcluded classes with < {min_samples_per_class}: "
              f"{len(color_counts) - len(valid_colors)}")
        
        filtered = [(p, c, m) for p, c, m in
                     zip(image_paths, colors, metadata) if c in valid_colors]
        image_paths = [x[0] for x in filtered]
        colors      = [x[1] for x in filtered]
        metadata    = [x[2] for x in filtered]

    unique_colors = sorted(set(colors))
    color_to_idx = {c: i for i, c in enumerate(unique_colors)}
    labels = [color_to_idx[c] for c in colors]

    print(f"\nClasses: {len(unique_colors)}: {unique_colors}")
    print(f"Total images count: {len(image_paths)}")

    return image_paths, labels, colors, metadata, color_to_idx, unique_colors

class CarColorDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
        except Exception:
            image = Image.new('RGB', (112, 112), color='black')

        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)

        return self.relu(out)

class CustomResNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.layer1 = self._make_layer(64,  64,  n_blocks=2, stride=1)
        self.layer2 = self._make_layer(64,  128, n_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, n_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(256, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(in_ch, out_ch, n_blocks, stride=1):
        layers = [ResBlock(in_ch, out_ch, stride)]
        
        for _ in range(1, n_blocks):
            layers.append(ResBlock(out_ch, out_ch, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)

        return self.fc(x)

def make_pretrained_resnet18(num_classes):
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes),
    )
    return model

IMG_SIZE = 112

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_transform_strong = transforms.Compose([
    transforms.Resize((IMG_SIZE + 24, IMG_SIZE + 24)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.25, contrast=0.25,
                           saturation=0.1, hue=0.02),
    transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
    transforms.RandomApply(
        [transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])