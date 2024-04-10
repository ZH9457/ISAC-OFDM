import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import commpy as cpy

# # a = np.array([1+1j, 2, 3, 4, 5])
# # b = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5])
# # c = np.vstack((a,a))
# # print(np.fft.fft(a),'\n',np.fft.fft(c, axis=1))
# # a = np.array([4, 8, 12, 16, 20])
# a = np.arange(16)
# b = [0, 1]
# c = np.array([8*b])
# # print(c)
# d = np.exp(1j*2*np.pi*c*(1/4))
# # a = np.exp(-1j*2*np.pi*a*2/16)
# # a = a*d
# print(abs(np.fft.ifft(d)))

# a = np.array([[1, 2, 3], [2, 3, 4]])
# b = np.array([[2, 3, 4], [1, 2, 3]])
# print(np.vsplit(a, [1])[0])

def swith(x):
    x[0] = 0
    # return x
a = np.array([1, 2])
b = [1, 1, 2, 2]
# swith(a)
print(np.fft.fft(a), '\n', np.fft.fft(b))
# print(a-1)