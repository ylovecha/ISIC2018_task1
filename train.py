import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import UNet
from dataset import ISIC2018Dataset
from utils import dice_coefficient, iou_score, pixel_accuracy, DiceBCELoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    for images, masks in tqdm(loader, desc="Training"):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_dice += dice_coefficient(outputs, masks)
        total_iou += iou_score(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # If validation ground truth is unavailable, split training data
    use_split_val = not os.path.exists(args.val_mask_dir)
    if use_split_val:
        print(f"Validation mask dir '{args.val_mask_dir}' not found — splitting training data (90/10).")
        # Create two datasets with the same files but different augmentation modes
        full_train_ds = ISIC2018Dataset(
            img_dir=args.train_img_dir, mask_dir=args.train_mask_dir,
            img_size=args.img_size, mode='train'
        )
        full_val_ds = ISIC2018Dataset(
            img_dir=args.train_img_dir, mask_dir=args.train_mask_dir,
            img_size=args.img_size, mode='val'
        )
        assert len(full_train_ds) == len(full_val_ds)
        n_total = len(full_train_ds)
        val_size = max(1, int(n_total * 0.1))
        train_size = n_total - val_size
        indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42)).tolist()
        train_dataset = torch.utils.data.Subset(full_train_ds, indices[:train_size])
        val_dataset = torch.utils.data.Subset(full_val_ds, indices[train_size:])
    else:
        train_dataset = ISIC2018Dataset(
            img_dir=args.train_img_dir,
            mask_dir=args.train_mask_dir,
            img_size=args.img_size,
            mode='train'
        )
        val_dataset = ISIC2018Dataset(
            img_dir=args.val_img_dir,
            mask_dir=args.val_mask_dir,
            img_size=args.img_size,
            mode='val'
        )

    num_workers = 2  # Use multiprocessing for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_dice = 0.0
    print(f"Training for {args.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice, val_iou = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Best model saved with Dice: {best_dice:.4f}")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(args.save_dir, 'last_model.pth'))

    print(f"\nTraining complete. Best Dice: {best_dice:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train U-Net for ISIC2018 Task1')
    parser.add_argument('--train_img_dir', type=str, default='data/ISIC2018_Task1-2_Training_Input')
    parser.add_argument('--train_mask_dir', type=str, default='data/ISIC2018_Task1_Training_GroundTruth')
    parser.add_argument('--val_img_dir', type=str, default='data/ISIC2018_Task1-2_Validation_Input')
    parser.add_argument('--val_mask_dir', type=str, default='data/ISIC2018_Task1_Validation_GroundTruth')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    main(args)
