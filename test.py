import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import sys
print(f"sys.version: {sys.version}")

try:
    import torch
    print(f"torch.__version__: {torch.__version__} and torch.cuda.is_available(): {torch.cuda.is_available()}")
except ImportError as e:
    print(f"Error importing torch: {e}")
try:
    import ultralytics
    print(f"ultralytics.__version__: {ultralytics.__version__}")
except ImportError as e:
    print(f"Error importing YOLO: {e}")

print(f"numpy.__version__: {np.__version__}")

array = np.random.rand(100, 100)
ndimage.gaussian_filter(array, sigma=3, output=array)
plt.imshow(array, cmap='viridis')
plt.colorbar()
plt.title('Random 100x100 Array')
plt.show()

