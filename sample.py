import numpy as np
def test(X):
    r = ([1,3,2])
    return X[r]

X = np.array([1,2,3,4,5])
X = test(X)
print(X)
