import numpy as np
import matplotlib.pyplot as plt

I = np.fromfile("data_34.35.bin", dtype = np.float32)

I = np.reshape(I, (1040, 1040, 165))

print(np.sum(np.sum(I, axis = 0), axis = 0))

#plt.matshow(np.sqrt(I[:, :, -1]))
#plt.show()