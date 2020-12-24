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

    def __str__(self):
        return self.description
    
    def psnr(self, img1, img2):
        im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
        im2 = cv2.imread(img2, cv2.IMREAD_COLOR)
        im1 = im1[0:im1.shape[0]-np.remainder(im1.shape[0], self.scale),
                  0:im1.shape[1]-np.remainder(im1.shape[1], self.scale),
                  :]
        print(im1.shape, im2.shape)
        pad = int( ( im1.shape[0] - ((im1.shape[0]-9+1) -5+1) ) / 2 )
        # compare only Y channel
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[pad:-pad, pad:-pad, 0]
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[pad:-pad, pad:-pad, 0]
        print(im1.shape, im2.shape)
        im1 = im1.astype('float')
        im2 = im2.astype('float')
        diff = (im1 - im2)**2.0
        psnr = 20*np.log10(255.0/np.sqrt(np.mean(diff)))
        return psnr
    
if __name__ == "__main__":
    obj = Utils()
    p0 = obj.psnr("../dataset/Test/Set5/butterfly_GT.bmp", "../result/butterfly_GT_bicubic.jpg")
    p1 = obj.psnr("../dataset/Test/Set5/butterfly_GT.bmp", "../result/butterfly_GT_srcnn.jpg")
    d = 100.0*(p1-p0)/p0
    print("butterfly_GT.bmp - cubic:%f  srcnn:%f  delta:%f %"%(p0, p1, d))

    p0 = obj.psnr("../dataset/Test/Set14/monarch.bmp", "../result/monarch_bicubic.jpg")
    p1 = obj.psnr("../dataset/Test/Set14/monarch.bmp", "../result/monarch_srcnn.jpg")
    d = 100.0*(p1-p0)/p0
    print("monarch.bmp      - cubic:%f  srcnn:%f  delta:%f %"%(p0, p1, d))

