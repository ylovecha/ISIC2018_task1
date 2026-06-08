import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model import UNet
from dataset import ISIC2018Dataset
from torch.utils.data import DataLoader
from utils import dice_coefficient, iou_score, pixel_accuracy


def predict_single(model, img_path, img_size, device):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    original_size = img.size
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output)
        pred = (pred > 0.5).float()

    pred_np = pred.squeeze().cpu().numpy()
    pred_img = Image.fromarray((pred_np * 255).astype(np.uint8))
    pred_img = pred_img.resize(original_size, Image.NEAREST)
    return pred_img


def evaluate_test(model, test_img_dir, test_mask_dir, img_size, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    test_dataset = ISIC2018Dataset(
        img_dir=test_img_dir,
        mask_dir=test_mask_dir,
        img_size=img_size,
        mode='test'
    )
    num_workers = 0 if os.name == 'nt' else 4
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=num_workers)

    model.eval()
    total_dice = 0
    total_iou = 0
    total_acc = 0
    count = 0

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            total_dice += dice_coefficient(outputs, masks)
            total_iou += iou_score(outputs, masks)
            total_acc += pixel_accuracy(outputs, masks)
            count += 1

    avg_dice = total_dice / count
    avg_iou = total_iou / count
    avg_acc = total_acc / count

    print(f"\nTest Results:")
    print(f"  Dice Coefficient: {avg_dice:.4f}")
    print(f"  IoU Score:        {avg_iou:.4f}")
    print(f"  Pixel Accuracy:   {avg_acc:.4f}")

    img_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(('.jpg', '.png'))])
    for img_name in tqdm(img_files[:20], desc="Saving predictions"):
        img_path = os.path.join(test_img_dir, img_name)
        pred_img = predict_single(model, img_path, img_size, device)
        base_name = os.path.splitext(img_name)[0]
        pred_img.save(os.path.join(output_dir, f"{base_name}_pred.png"))

    return avg_dice, avg_iou, avg_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.model_path}")

    if args.single_image:
        pred_img = predict_single(model, args.single_image, args.img_size, device)
        output_path = args.single_image.replace('.jpg', '_pred.png').replace('.png', '_pred.png')
        pred_img.save(output_path)
        print(f"Prediction saved to {output_path}")
    else:
        evaluate_test(model, args.test_img_dir, args.test_mask_dir, args.img_size, device, args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with U-Net for ISIC2018 Task1')
    parser.add_argument('--model_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--test_img_dir', type=str, default='data/ISIC2018_Task1-2_Test_Input')
    parser.add_argument('--test_mask_dir', type=str, default='data/ISIC2018_Task1_Test_GroundTruth')
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--output_dir', type=str, default='predictions')
    parser.add_argument('--single_image', type=str, default=None)
    args = parser.parse_args()
    main(args)
