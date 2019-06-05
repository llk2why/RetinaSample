import cv2
import numpy as np

x = cv2.imread('x.png')
y = cv2.imread('y.png')

cv2.imshow('x',x)
cv2.imshow('y',y)
cv2.waitKey(0)

print(x.dtype)
z = np.zeros_like(x)
print(np.max(x),np.max(y))
def f(x,y):
    mse = np.mean(np.square(x.astype(np.float)-y))
    psnr = 20*np.log10(255)-10*np.log10(mse)
    return psnr
a = f(x,y)
b = f(y+0.01,y)
c = f(z,y)
print(a,b,c)