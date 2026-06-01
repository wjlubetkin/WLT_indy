"""
Fine-tune MegaDescriptor-S-224 on a folder-structured wildlife dataset.
Expected dataset layout:
    <data_dir>/
        <identity_1>/
            image_1.jpg
            image_2.jpg
        <identity_2>/
            ...
Usage:
    python train.py --data_dir test_20 --output_dir runs/exp1
"""

import argparse
import os
import pandas as pd
import timm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import transforms

from wildlife_tools.data import ImageDataset
from wildlife_tools.train import BasicTrainer
from wildlife_tools.train.objective import ArcFaceLoss
from wildlife_tools.train.callbacks import EpochCheckpoint


EMBEDDING_SIZE = 768  # swin_small output dim


def build_metadata(data_dir: str) -> pd.DataFrame:
    records = []
    for identity in sorted(os.listdir(data_dir)):
        identity_dir = os.path.join(data_dir, identity)
        if not os.path.isdir(identity_dir):
            continue
        for fname in sorted(os.listdir(identity_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                records.append({
                    "path": os.path.join(identity_dir, fname),
                    "identity": identity,
                })
    df = pd.DataFrame(records)
    print(f"Dataset: {len(df)} images, {df['identity'].nunique()} identities")
    return df


def get_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = build_metadata(args.data_dir)
    dataset = ImageDataset(metadata, transform=get_transforms())

    backbone = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-S-224",
        num_classes=0,
        pretrained=True,
    )

    objective = ArcFaceLoss(
        num_classes=dataset.num_classes,
        embedding_size=EMBEDDING_SIZE,
        margin=args.margin,
        scale=args.scale,
    )

    optimizer = optim.SGD(
        list(backbone.parameters()) + list(objective.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    checkpoint_cb = EpochCheckpoint(folder=args.output_dir, save_step=args.save_every)

    trainer = BasicTrainer(
        dataset=dataset,
        model=backbone,
        objective=objective,
        optimizer=optimizer,
        epochs=args.epochs,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        accumulation_steps=args.accumulation_steps,
        epoch_callback=checkpoint_cb,
    )

    trainer.train()

    # Save final model weights separately for easy inference loading
    final_path = os.path.join(args.output_dir, "megadescriptor_s224_finetuned.pt")
    torch.save(backbone.state_dict(), final_path)
    print(f"Saved final weights to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="test_20",
                        help="Root folder of the dataset (identity sub-dirs)")
    parser.add_argument("--output_dir", default="runs/exp1",
                        help="Where to save checkpoints and final weights")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # ArcFace hyperparameters
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scale", type=int, default=64)

    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save a checkpoint every N epochs")

    main(parser.parse_args())
