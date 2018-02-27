import numpy as np
import cv2

########Q2########

a = np.matrix('1,2;2,2;2,1')
VSU = np.linalg.svd(a)

########Q3########

img = cv2.imread('Cleto2.jpeg',0)

print(img)

cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

svd = np.linalg.svd(img, full_matrices=False)

# I = svd[1]*np.identity(768)
# Z = np.zeros([768,226])

U = svd[0]
# S = np.concatenate((I,Z),axis=1)
S = np.diag(svd[1])
V = svd[2]

r = 9

u = U[:,0:r]
s = S[0:r,0:r]
v = V[0:r,:]

us = np.dot(u,s)

A = np.dot(us,v)

np.allclose(img, A)

im = np.array(A, dtype = np.uint8)

# A = np.dot(U,np.dot(S,V))

cv2.imshow('image2',im)
cv2.waitKey(0)
cv2.destroyAllWindows()