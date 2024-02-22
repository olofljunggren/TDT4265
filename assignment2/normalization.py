import numpy as np
from utils import *

X_train, Y_train, *_ = load_full_mnist()


mean = np.mean(X_train)
std = np.std(X_train)

print("Dataset mean:", mean)
print("Dataset standard deviation:", std)
