import os
import sys
import torch
from thop import profile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from models.cnn import QuantizableCNN

model = QuantizableCNN().eval()
dummy = torch.randn(1, 3, 32, 32)

macs, params = profile(model, inputs=(dummy,), verbose=False)

print("========== MODEL COMPLEXITY ==========")
print(f"MACs       : {macs:,.0f}")
print(f"MACs       : {macs / 1e6:.2f} MMAC")
print(f"Parameters : {params:,.0f}")
print("======================================")
