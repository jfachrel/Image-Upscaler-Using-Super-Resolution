import cv2
import matplotlib.pyplot as plt
import os

from cv2 import dnn_superres

class SuperResolution():
    def __init__(self):
        self.image = None
        self.face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_eye.xml')

    def cropped(self, image):
        # image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(image, 1.3,5)
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_color)
            if len(eyes) >= 2:
                return roi_color

    def resizing(self, image):
        self.image = image
        image = self.cropped(image)
        w,h,_ = image.shape
        up_width = w*4
        up_height = h*4
        up_points = (up_width, up_height)
        resized_up = cv2.resize(image, up_points, interpolation= cv2.INTER_LINEAR)

        return resized_up
    def resolution(self,image,model,base_path='models'):
        self.image = image
        image = self.cropped(image)
        sr = dnn_superres.DnnSuperResImpl_create()
        model_path = os.path.join(base_path, model +".pb")
        model_name = model.split('_')[0].lower()
        model_scale = int(model.split("_")[1][1])
        sr.readModel(model_path)
        sr.setModel(model_name, model_scale)
        Final_Img = sr.upsample(image)

        return Final_Img