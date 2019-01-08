import cv2
import numpy as np
import cvxpy as cp
import math
import sys

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    MAX = 255.0
    return 10*math.log10(MAX*MAX/mse)

def kernel2matrixA(kernel, height, width):
    N = height * width
    k_size = kernel.shape[0]
    c = math.floor(k_size/2)
    
    A = np.zeros((N, N))
    for col in range(height):
        for row in range(width):
            tmp = np.zeros((height, width))
            for j in range(-c, c+1):
                for i in range(-c, c+1):
                    y = cv2.borderInterpolate(col+j, height, cv2.BORDER_REPLICATE)
                    x = cv2.borderInterpolate(row+i, width, cv2.BORDER_REPLICATE)
                    tmp[y, x] += kernel[j+c, i+c]
            A[col*width+row] = np.reshape(tmp, N)
    return A

def main():
    img_src = cv2.imread("birds_gray.png", 0)
    img_src = cv2.resize(img_src, None, fx=0.2, fy=0.2)
    height, width = img_src.shape
    N = height*width
    
    #3*3 Gaussian filter
    kernel = np.array([[0.00854167, 0.02230825, 0.03072131, 0.02230825, 0.00854167],
                       [0.02230825, 0.05826239, 0.08023475, 0.05826239, 0.02230825],
                       [0.03072131, 0.08023475, 0.11049350, 0.08023475, 0.03072131],
                       [0.00854167, 0.02230825, 0.03072131, 0.02230825, 0.00854167],
                       [0.02230825, 0.05826239, 0.08023475, 0.05826239, 0.02230825]])

    A = kernel2matrixA(kernel, height, width)
    
    x_src = np.reshape(img_src, N)
    sigma = 5
    n = np.random.normal(0, sigma, N)
    #b = Ax + n
    b = np.clip(np.dot(A, x_src) + n, 0, 255)
    img_blur_f = np.reshape(b, (height, width))
    img_blur = img_blur_f.astype(np.uint8)

    print("image size", height, width)
    print("sigma", sigma)
    
    print("sparse optimisation")
    Lambda = 0.5
    print("lambda", Lambda)
    
    x = cp.Variable(N)
    obj = cp.Minimize(cp.sum_squares(b-A*x)/2 + Lambda*cp.tv(cp.reshape(x,(height,width))))
    constraints = [0<=x, x<=255]
    prob = cp.Problem(obj, constraints) 
    prob.solve()
    
    img_dst = np.reshape(np.clip(x.value, 0, 255), (height, width)).astype(np.uint8)
    
    print("blur PSNR", psnr(img_src, img_blur_f), "[dB]")
    print("dst PSNR", psnr(img_src, img_dst), "[dB]")
    
    cv2.imshow("src", img_src)
    cv2.imshow("blur", img_blur)
    cv2.imshow("dst", img_dst)
    cv2.imwrite("result/src.png", img_src)
    cv2.imwrite("result/blur.png", img_blur) 
    cv2.imwrite("result/deblur.png", img_dst)
    cv2.waitKey(0)

if __name__=='__main__':
    sys.exit(main())
