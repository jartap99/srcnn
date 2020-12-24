#
# @rajeevp
#

import numpy as np
import cv2

class Utils(object):
    def __init__(self):
        np.random.seed(9076301)
        self.description = " Utilities"
        self.scale=3
        self.img_orig = None
        self.img_cubic = None
        self.img_srcnn = None
        self.img_srcnn_np = None
        self.pad = None

    def __str__(self):
        return self.description
    
    def ssim(self, im1, im2):
        mean1 = np.mean(im1)
        mean2 = np.mean(im2)
        var1 = np.var(im1)
        var2 = np.var(im2)
        covar = np.mean((im1-mean1)*(im2-mean2))
        c1 = np.square(0.01 * 255)
        c2 = np.square(0.03 * 255)
        num = (2 * mean1 * mean2 + c1) * (2 * covar + c2)
        denom = (mean1**2 + mean2**2 + c1) * (var1 + var2 + c2)
        val = num / denom
        return val


    def psnr(self, im1, im2):
        diff = (im1 - im2)**2.0
        val = 20*np.log10(255.0/np.sqrt(np.mean(diff)))
        return val
    
    def compare_images(self, img_orig, img_cubic, img_srcnn, img_srcnn_np):
        print("***** compare images *****")
        im = cv2.imread(img_orig, cv2.IMREAD_COLOR)
        im = im[0:im.shape[0]-np.remainder(im.shape[0], self.scale),
                0:im.shape[1]-np.remainder(im.shape[1], self.scale),
                :]
        self.pad = int( ( im.shape[0] - ((im.shape[0]-9+1) -5+1) ) / 2 )
        self.img_orig = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[self.pad:-self.pad, self.pad:-self.pad, 0]
        self.img_orig = self.img_orig.astype('float')

        im = cv2.imread(img_cubic, cv2.IMREAD_COLOR)
        self.img_cubic = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[self.pad:-self.pad, self.pad:-self.pad, 0]
        self.img_cubic = self.img_cubic.astype('float')

        im = cv2.imread(img_srcnn, cv2.IMREAD_COLOR)
        self.img_srcnn = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[self.pad:-self.pad, self.pad:-self.pad, 0]
        self.img_srcnn = self.img_srcnn.astype('float')

        im = cv2.imread(img_srcnn, cv2.IMREAD_COLOR)
        self.img_srcnn_np = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)[self.pad:-self.pad, self.pad:-self.pad, 0]
        self.img_srcnn_np = self.img_srcnn_np.astype('float')

        # Compare PSNR
        p0 = self.psnr(self.img_orig, self.img_cubic)
        p1 = self.psnr(self.img_orig, self.img_srcnn)
        p2 = self.psnr(self.img_orig, self.img_srcnn_np)
        d0 = 100.0*(p1-p0)/p0
        d1 = 100.0*(p2-p0)/p0
        print(img_orig.split("/")[-1], "Dimensions:", self.img_orig.shape)
        print("PSNR cubic: %f  srcnn   : %f  Improvement: %f percent"%(p0, p1, d0))
        print("PSNR cubic: %f  srcnn_np: %f  Improvement: %f percent"%(p0, p2, d1))

        # Compare SSIM
        p0 = self.ssim(self.img_orig, self.img_cubic)
        p1 = self.ssim(self.img_orig, self.img_srcnn)
        p2 = self.ssim(self.img_orig, self.img_srcnn_np)
        print("SSIM cubic: %f  srcnn   : %f  delta: %f"%(p0, p1, p1-p0))
        print("SSIM cubic: %f  srcnn_np: %f  delta: %f"%(p0, p2, p2-p0))

        # Compare MSE
        p0 = np.mean(np.square(self.img_orig - self.img_cubic))
        p1 = np.mean(np.square(self.img_orig - self.img_srcnn))
        p2 = np.mean(np.square(self.img_orig - self.img_srcnn_np))
        d0 = 100.0*(p1-p0)/p0
        d1 = 100.0*(p2-p0)/p0
        print("MSE cubic: %f  srcnn   : %f  Delta: %f percent"%(p0, p1, d0))
        print("MSE cubic: %f  srcnn_np: %f  Delta: %f percent"%(p0, p2, d1))


    
if __name__ == "__main__":
    obj = Utils()
    obj.compare_images( img_orig="../dataset/Test/Set5/butterfly_GT.bmp", 
                        img_cubic="./data/butterfly_GT_cubic.jpg",
                        img_srcnn="./data/butterfly_GT_srcnn.jpg",
                        img_srcnn_np="./data/butterfly_GT_srcnn_np.jpg")
    obj.compare_images( img_orig="../dataset/Test/Set14/monarch.bmp", 
                        img_cubic="./data/monarch_cubic.jpg",
                        img_srcnn="./data/monarch_srcnn.jpg",
                        img_srcnn_np="./data/monarch_srcnn_np.jpg")
    obj.compare_images( img_orig="../dataset/Test/Set14/everest.jpg", 
                        img_cubic="./data/everest_cubic.jpg",
                        img_srcnn="./data/everest_srcnn.jpg",
                        img_srcnn_np="./data/everest_srcnn_np.jpg")
    