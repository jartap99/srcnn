#
# @rajeevp
#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from dnn import *

class SRCNN(DNN):
    def __init__(self, weightsFile):
        DNN.__init__(self)
        np.random.seed(9076301)
        print("tf.__version__: ", tf.__version__)
        self.weightsFile = weightsFile
        self.model = "Not yet initialized"
        self.load_model()
        self.scale = 3
        self.hr_img = None
        self.lr_img = None
        self.sr_img = None
        self.layers = ['layer1', 'layer2', 'layer3']
        self.inputs = None
        self.ofms = {}
        self.weights = {}
        self.biases = {}

    def __str__(self):
        return str(self.model.summary()) + "\n*** Initialized weights with %s ***\n"%self.weightsFile
    
    def load_model(self):
        inputs  = tf.keras.Input(shape=(None, None, 1))
        x       = tf.keras.layers.Conv2D(filters=64, kernel_size=9, padding="valid", activation="relu", name="layer1")(inputs)
        x       = tf.keras.layers.Conv2D(filters=32, kernel_size=1, padding="same",  activation="relu", name="layer2")(x)
        outputs = tf.keras.layers.Conv2D(filters=1,  kernel_size=5, padding="valid", name="layer3")(x)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="srcnn")
        self.model.compile( optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001),
                            loss=tf.keras.losses.MeanSquaredError(),
                            #metrics=[compute_psnr_keras],
                            metrics=['mse']  )
        self.model.load_weights(self.weightsFile)
        
    def pre_process(self, path, imageFile):
        self.hr_img = cv2.cvtColor( cv2.imread(path + imageFile, cv2.IMREAD_COLOR), cv2.COLOR_BGR2YCrCb)
        self.hr_img = self.hr_img[  0:self.hr_img.shape[0]-np.remainder(self.hr_img.shape[0],self.scale),
                                    0:self.hr_img.shape[1]-np.remainder(self.hr_img.shape[1],self.scale),
                                    : ]
        self.lr_img = cv2.resize(self.hr_img, None, fx=1/self.scale, fy=1/self.scale, interpolation=cv2.INTER_CUBIC)
        self.lr_img = cv2.resize(self.lr_img, None, fx=self.scale/1, fy=self.scale/1, interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(self.lr_img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite("./data/" + imageFile.split(".")[0] + "_cubic.jpg", img)
        print("hr_img.shape : ", self.hr_img.shape)
        print("lr_img.shape : ", self.lr_img.shape)

    def enhance(self, path, imageFile, saveTemps=False):
        self.pre_process(path, imageFile)
        lr_img = self.lr_img[:,:,0].astype('float')/255.0
        ifm = lr_img.reshape(1, lr_img.shape[0], lr_img.shape[1], 1)
        predicted = self.model.predict( ifm)
        predicted = predicted*255.0
        predicted[predicted[:] > 255]  = 255
        predicted[predicted[:] < 0]    = 0
        self.sr_img = self.lr_img
        self.pad = int( ( self.sr_img.shape[0] - (self.sr_img.shape[0]-9+1 -5+1) ) /2)
        self.sr_img[self.pad:-self.pad, self.pad:-self.pad, 0] = predicted[0,:,:,0].astype('uint8')
        img = cv2.cvtColor( self.sr_img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite("./data/" + imageFile.split(".")[0] + "_srcnn.jpg", img)
        print("sr_img.shape : ", self.sr_img.shape)
        print("Image enhanced ...")
        print("saveTemps : ", saveTemps)
        if (saveTemps==True):
            for layerName in self.layers+['inputs']:
                self.extract_features(ifm=ifm, layerName=layerName)
            for layerName in self.layers:
                self.extract_weights(layerName=layerName)
        
        # usually numpy implemented conv2D + ReLu
        self.enhance_np(imageFile)     

    def extract_features(self, ifm, layerName):
        if layerName=="inputs":
            ifm = ifm[0,:,:,0]
            self.inputs = ifm.reshape(ifm.shape[0], ifm.shape[1], 1)
            fw = open("./data/" + "inputs.txt", "w")
            for i in range(ifm.shape[0]):
                for j in range(ifm.shape[1]):
                    fw.write("%f\n"%ifm[i,j])
            fw.close()
        else:
            subModel = tf.keras.models.Model(inputs=self.model.inputs, 
                                             outputs=self.model.get_layer(name=layerName).output)
            ofm = subModel.predict(ifm)[0,:,:,:]
            self.ofms[layerName] = ofm
            print("ofm - ", layerName, "  shape:", ofm.shape)
            fw = open("./data/" + layerName + "_ofm.txt", "w")
            #for k in range(ofm.shape[2]):
            for k in range(1):
                for i in range(ofm.shape[0]):
                    for j in range(ofm.shape[1]):
                        fw.write("%f\n"%ofm[i,j,k])
            fw.close()

    def extract_weights(self, layerName):
        weights, biases = self.model.get_layer(name=layerName).get_weights()
        print("weights : ", weights.shape)
        print("biases  : ", biases.shape)
        self.weights[layerName] = weights
        self.biases[layerName] = biases
        fw = open("./data/" + layerName + "_weights.txt", "w")
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                fw.write("%f\n"%weights[i,j,0,0])
        fw.write("%f\n"%biases[0])
        fw.close()

    def enhance_np(self, imageFile):
        ofm1 = DNN.conv2D(  ifm=self.inputs,
                            weights=self.weights['layer1'],
                            biases=self.biases['layer1'],
                            padding="valid",
                            activation="relu" )
        diff = ofm1 - self.ofms['layer1']
        print("Layer 1 diff: ", np.sum(diff), diff.shape)

        ofm2 = DNN.conv2D(  ifm=ofm1,
                            weights=self.weights['layer2'],
                            biases=self.biases['layer2'],
                            padding="same",
                            activation="relu" )
        diff = ofm2 - self.ofms['layer2']
        print("Layer 2 diff: ", np.sum(diff), diff.shape)

        ofm3 = DNN.conv2D(  ifm=ofm2,
                            weights=self.weights['layer3'],
                            biases=self.biases['layer3'],
                            padding="valid",
                            activation="none" )
        diff = ofm3 - self.ofms['layer3']
        print("Layer 3 diff: ", np.sum(diff), diff.shape)

        ofm3 = ofm3*255.0
        ofm3[ofm3[:]>255] = 255
        ofm3[ofm3[:]<0]   = 0
        sr_img1 = self.lr_img
        sr_img1[self.pad:-self.pad, self.pad:-self.pad, 0] = ofm3[:,:,0].astype('uint8')
        img = cv2.cvtColor(sr_img1, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite("./data/" + imageFile.split(".")[0] + "_srcnn_np.jpg", img)
                         
if __name__ == "__main__":
    srcnn = SRCNN("../metric_mse/srcnn_915_lr0.0001_epochs300-400.h5")
    print(srcnn)
    if False:
        srcnn.enhance(  path="../dataset/Test/Set5/", 
                        imageFile="butterfly_GT.bmp",
                        saveTemps=True)
        srcnn.enhance(  path="../dataset/Test/Set14/", 
                        imageFile="monarch.bmp",
                        saveTemps=True)
    srcnn.enhance(  path="../dataset/Test/Set14/", 
                    imageFile="everest.jpg",
                    saveTemps=True)

        