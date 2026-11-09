import os
import shutil
from pathlib import Path

os.chdir(r'C:\Users\mohit\OneDrive\Documents\bsc data scince\python\project\pothole-detection')

# Copy epoch10.pt over best.pt so extend mode starts from the better weights
best_pt = Path('runs/pothole/train_v3/weights/best.pt')
epoch10_pt = Path('runs/pothole/train_v3/weights/epoch10.pt')

shutil.copy2(epoch10_pt, best_pt)
print(f'Copied epoch10.pt -> best.pt')
print(f'  {best_pt}')
print(f'  Size: {best_pt.stat().st_size / (1024*1024):.1f} MB')
print()

# Verify
import hashlib
def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

print(f'  best.pt md5 after copy: {md5(best_pt)}')
print(f'  epoch10.pt md5:         {md5(epoch10_pt)}')
print('  Match: verified' if md5(best_pt) == md5(epoch10_pt) else '  MISMATCH!')
