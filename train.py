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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import transforms

from wildlife_tools.data import ImageDataset
from wildlife_tools.train.objective import ArcFaceLoss


EMBEDDING_SIZE = 768  # swin_small output dim


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

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


def split_metadata(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    # Drop identities with only 1 image (can't stratify or evaluate)
    counts = df["identity"].value_counts()
    df = df[df["identity"].isin(counts[counts > 1].index)].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["identity"], random_state=seed
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_train_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
        transforms.RandAugment(num_ops=2, magnitude=20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_eval_transforms(size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, objective, optimizer, loader, device, accumulation_steps):
    model.train()
    objective.train()
    optimizer.zero_grad()
    total_loss = 0.0

    for step, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)
        loss = objective(embeddings, labels) / accumulation_steps
        loss.backward()
        total_loss += loss.item() * accumulation_steps

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Evaluation — rank-1 accuracy via cosine nearest-neighbour
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_embeddings(model, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False)
    model.eval()
    all_embs, all_labels = [], []
    for images, labels in loader:
        embs = model(images.to(device))
        embs = F.normalize(embs, dim=1)
        all_embs.append(embs.cpu())
        all_labels.append(labels)
    return torch.cat(all_embs), torch.cat(all_labels)


def rank1_accuracy(embeddings, labels):
    """Leave-one-out rank-1: for each sample, find its nearest neighbour
    (excluding itself) and check if it shares the same identity."""
    sim = embeddings @ embeddings.T          # cosine similarity matrix
    sim.fill_diagonal_(-float("inf"))        # exclude self
    nn_idx = sim.argmax(dim=1)
    correct = (labels[nn_idx] == labels).float().mean().item()
    return correct


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(train_losses, test_accuracies, output_dir):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(epochs, train_losses, color="steelblue", linewidth=2)
    ax1.set_ylabel("Train Loss (ArcFace)")
    ax1.set_title("Training Loss vs Epoch")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in test_accuracies],
             color="darkorange", linewidth=2)
    ax2.set_ylabel("Rank-1 Accuracy (%)")
    ax2.set_xlabel("Epoch")
    ax2.set_title("Test Rank-1 Accuracy vs Epoch")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = build_metadata(args.data_dir)
    train_df, test_df = split_metadata(metadata, test_size=args.test_size)
    print(f"Train: {len(train_df)} images | Test: {len(test_df)} images")

    train_dataset = ImageDataset(train_df, transform=get_train_transforms())
    test_dataset  = ImageDataset(test_df,  transform=get_eval_transforms())

    backbone = timm.create_model(
        "hf-hub:BVRA/MegaDescriptor-S-224",
        num_classes=0,
        pretrained=True,
    ).to(args.device)

    objective = ArcFaceLoss(
        num_classes=train_dataset.num_classes,
        embedding_size=EMBEDDING_SIZE,
        margin=args.margin,
        scale=args.scale,
    ).to(args.device)

    optimizer = optim.SGD(
        list(backbone.parameters()) + list(objective.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )

    train_losses, test_accuracies = [], []
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            backbone, objective, optimizer, train_loader,
            args.device, args.accumulation_steps,
        )
        scheduler.step()

        embeddings, labels = extract_embeddings(
            backbone, test_dataset, args.batch_size, args.num_workers, args.device
        )
        acc = rank1_accuracy(embeddings, labels)

        train_losses.append(loss)
        test_accuracies.append(acc)

        if acc > best_acc:
            best_acc, best_epoch = acc, epoch
            torch.save(backbone.state_dict(),
                       os.path.join(args.output_dir, "best_model.pt"))

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"loss={loss:.4f}  "
              f"rank1={acc*100:.1f}%  "
              f"(best={best_acc*100:.1f}% @ epoch {best_epoch})")

        if epoch % args.save_every == 0:
            torch.save(backbone.state_dict(),
                       os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))

    # Final summary
    print("\n" + "=" * 50)
    print(f"Training complete.")
    print(f"Best rank-1 accuracy: {best_acc*100:.1f}% (epoch {best_epoch})")
    print("=" * 50)

    torch.save(backbone.state_dict(),
               os.path.join(args.output_dir, "megadescriptor_s224_finetuned.pt"))

    save_plot(train_losses, test_accuracies, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="test_20")
    parser.add_argument("--output_dir", default="runs/exp1")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of images per identity held out for testing")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # ArcFace
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scale", type=int, default=64)

    # System
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=10)

    main(parser.parse_args())
