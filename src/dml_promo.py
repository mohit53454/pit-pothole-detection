"""Test DirectML type promotion for float*bool (in-place and out-of-place)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
import dml_compat

dml = dml_compat.install()

a = torch.tensor([1.5, 2.5, 3.5, 4.5], device=dml)
m = torch.tensor([True, False, True, True], device=dml)

print("CPU  out-place:", ((a.cpu() * m.cpu())).tolist(), (a.cpu() * m.cpu()).dtype)
print("DML  out-place dtype:", (a * m).dtype, "values:", (a * m).cpu().tolist())
print("DML  torch.mul dtype:", torch.mul(a, m).dtype, "values:", torch.mul(a, m).cpu().tolist())

b = a.clone()
ptr_before = b.data_ptr()
b *= m
print("DML  imul  dtype:", b.dtype, "values:", b.cpu().tolist(), "same_storage:", b.data_ptr() == ptr_before)

c = a.clone()
r = c.mul_(m)
print("DML  mul_  dtype:", r.dtype, "values:", r.cpu().tolist(), "same_storage:", r.data_ptr() == c.data_ptr(),
      "is_self:", r is c)

# bool*float and bool*bool sanity
mb = m.clone()
try:
    mb2 = mb * a
    print("DML  bool*float dtype:", mb2.dtype, "values:", mb2.cpu().tolist())
except Exception as e:
    print("DML  bool*float raised:", e)

print("PROMO_DONE")
