import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ISIC2018Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=256, mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

        if mode == 'train':
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20),
                transforms.ToTensor()
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.mask_transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)

        base_name = os.path.splitext(img_name)[0]
        mask_candidates = [
            os.path.join(self.mask_dir, base_name + '_segmentation.png'),
            os.path.join(self.mask_dir, base_name + '.png'),
            os.path.join(self.mask_dir, base_name + '_segmentation.jpg'),
        ]

        mask_path = None
        for candidate in mask_candidates:
            if os.path.exists(candidate):
                mask_path = candidate
                break

        img = Image.open(img_path).convert('RGB')

        if mask_path is not None:
            mask = Image.open(mask_path).convert('L')
        else:
            mask = Image.new('L', img.size, 0)

        seed = np.random.randint(2147483647)

        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        img = self.img_transform(img)

        np.random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()

        return img, mask
