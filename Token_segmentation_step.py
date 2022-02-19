import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from skimage.io import imread_collection,imsave
import skimage.io as io
import shutil
import tensorflow as tf
from keras import models, layers, optimizers, regularizers
from keras.applications import VGG16, VGG19, ResNet50, Xception
from keras.applications.densenet import DenseNet121
from sklearn.metrics import roc_auc_score
from keras.models import load_model, Model
import pickle
import glob
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.applications.vgg16 import preprocess_input
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import matplotlib.pyplot as plt
from cv2 import imread, inpaint, imshow, INPAINT_TELEA
from PIL import Image

path_model = '/root/codigos_luis/Seg_Marcacao_classifier.h5'
model = load_model(path_model)

arr_teste_anormal = os.listdir('/dados/databases/ATUALIZACAO_4_IA/set_dez_processadas')
pth_teste_anormal = '/dados/databases/ATUALIZACAO_4_IA/set_dez_processadas/'

tamanho=(256, 256)
   
path_save = '/dados/databases/ATUALIZACAO_4_IA/set_dez_processadas_seg/'   

for n in arr_teste_anormal:
    if n == '.ipynb_checkpoints':
        continue
        
    if os.path.exists(path_save + n):
        continue

    path_img = pth_teste_anormal+n
    
    img = image.load_img(path_img)
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)

    predictions_ = model.predict(img_data)
    
    img = array_to_img(predictions_[0])
    images = np.asanyarray(img)
    
    original = cv2.imread(path_img)
    original = original.astype('uint8')
    open_cv_image = np.array(img) 
    
    result = cv2.inpaint(original,open_cv_image,3,INPAINT_TELEA)
    
    io.imsave(path_save + n,result)
    