# This is a sample Python script.

import pandas as pd
import numpy as np

a = np.array([1, 2, 3])
b = np.array([5, 1, 2])
b[0] = 4
image = np.array([[5, 1, 9, 3, 3],
                  [7, 9, 2, 5, 7],
                  [1, 7, 2, 8, 6],
                  [3, 6, 4, 8, 1],
                  [4, 9, 3, 3, 6]])

print(b)
c = a - b
print(c**2)
print(c)
print(np.exp(c))