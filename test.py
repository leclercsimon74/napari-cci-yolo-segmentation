import matplotlib.pyplot as plt
import numpy as np

array = np.random.rand(10, 10)
plt.imshow(array, cmap='viridis')
plt.colorbar()
plt.title('Random 10x10 Array')
plt.show()
