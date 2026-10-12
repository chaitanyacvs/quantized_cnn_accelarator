import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.cnn import QuantizableCNN

model = QuantizableCNN()

total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("========== MODEL INFORMATION ==========")
print(f"Total parameters     : {total:,}")
print(f"Trainable parameters : {trainable:,}")
print(f"FP32 parameter size  : {total * 4 / 1024:.2f} KB")
print(f"INT8 parameter size  : {total / 1024:.2f} KB")
print("========================================")
