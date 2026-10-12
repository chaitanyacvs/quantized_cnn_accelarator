import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cnn import CompactCNN


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = "./data"
MODEL_PATH = "./results/cnn_fp32.pth"

BATCH_SIZE = 128
NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 60)
print("FP32 MODEL EVALUATION")
print("=" * 60)
print(f"Device: {device}")


# ============================================================
# TRANSFORM
# ============================================================

test_transform = transforms.Compose([
    transforms.ToTensor(),

    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616)
    )
])


# ============================================================
# DATASET
# ============================================================

test_dataset = datasets.CIFAR10(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=test_transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)


# ============================================================
# MODEL
# ============================================================

model = CompactCNN(num_classes=10)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Your train.py saves a dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    saved_accuracy = checkpoint.get(
        "accuracy",
        None
    )

    saved_epoch = checkpoint.get(
        "epoch",
        None
    )

else:
    # Also support a raw state_dict
    model.load_state_dict(checkpoint)

    saved_accuracy = None
    saved_epoch = None


model = model.to(device)
model.eval()


# ============================================================
# EVALUATION
# ============================================================

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100.0 * correct / total


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Test Accuracy : {accuracy:.2f}%")

if saved_accuracy is not None:
    print(
        f"Saved Best Accuracy : "
        f"{saved_accuracy:.2f}%"
    )

if saved_epoch is not None:
    print(
        f"Best Epoch : {saved_epoch}"
    )

print("=" * 60)
