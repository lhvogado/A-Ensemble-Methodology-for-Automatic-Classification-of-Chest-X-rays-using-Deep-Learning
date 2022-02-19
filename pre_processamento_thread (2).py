import glob
import os
import math
import time
import numpy as np
import skimage.io as io
from skimage import morphology
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure, img_as_ubyte
from skimage.filters import threshold_otsu
import threading
import subprocess

def pre_processamento_transformar_imagens_em_quadradas_redimensionar_paralelo(i, starting_at, names, path_save, 
                                                                              size1=256, size2=256, radius=5):
    print(i,starting_at)
    for name in names:
        try:
            if os.path.exists(path_save + name.split('/')[4]):
                continue
                
            im = io.imread(name)
            
            thresh = threshold_otsu(im)
            binary = im > thresh

            #Abertura morfologia para eliminar pixels isolados
            selem = morphology.disk(radius,dtype=np.bool)
            aux = morphology.binary_opening(binary,selem)

            #Pegando os limites inferiores e superiores das regioes com informacao na imagem
            l_min = np.min(aux.nonzero()[0])
            l_max = np.max(aux.nonzero()[0])
            c_min = np.min(aux.nonzero()[1])
            c_max = np.max(aux.nonzero()[1])

            #Verificando qual a maior dimensao resultante, se linhas ou colunas
            dif1 = l_max - l_min
            dif2 = c_max - c_min

            if(dif1 > dif2):
                imagem_quadrada = np.zeros((dif1,dif1),im.dtype)
                imagem_quadrada[:,:] = im.min()
                intervalo1 = math.ceil((dif1-dif2)/2)
                intervalo2 = math.floor((dif1-dif2)/2)

                if(c_min - intervalo1 > 0):
                    if(c_max + intervalo2 <= im.shape[1]):
                        imagem_quadrada[0:,0:] = im[l_min:l_max,c_min-intervalo1:c_max+intervalo2]
                    else:
                        imagem_quadrada[0:,0:im.shape[1]] = im[l_min:l_max,0:imagem_quadrada.shape[1]]
                else:
                    if(im.shape[1] < imagem_quadrada.shape[1]):
                        imagem_quadrada[0:,0:im.shape[1]] = im[l_min:l_max,0:]
                    else:
                        imagem_quadrada[0:,0:] = im[l_min:l_max,0:imagem_quadrada.shape[1]]

            else: #Se entrar aqui eh pq dif2 eh maior
                imagem_quadrada = np.zeros((dif2,dif2),im.dtype)
                imagem_quadrada[:,:] = im.min()
                intervalo1 = math.ceil((dif2-dif1)/2)
                intervalo2 = math.floor((dif2-dif1)/2)

                if(l_min - intervalo1 > 0):
                    if(l_max + intervalo2 <= im.shape[0]):
                        #print('caso 3')
                        imagem_quadrada[0:,0:] = im[l_min-intervalo1:l_max+intervalo2,c_min:c_max]
                    else:
                        if(imagem_quadrada.shape[0] > im.shape[0]):
                            imagem_quadrada[0:im.shape[0],0:] = im[0:im.shape[0],c_min:c_max]
                        else:
                            imagem_quadrada[0:imagem_quadrada.shape[0],0:] = im[0:imagem_quadrada.shape[0],c_min:c_max]

                else:
                    if(im.shape[0] < imagem_quadrada.shape[0]):
                        imagem_quadrada[0:im.shape[0],0:] = im[0:,c_min:c_max]
                    else:
                        imagem_quadrada[0:,0:] = im[0:imagem_quadrada.shape[0],c_min:c_max]

            if(np.max(imagem_quadrada) > 10):
                im_resized = resize(imagem_quadrada, (size1, size2),preserve_range=True,mode='edge')
                im_resized = np.uint16(im_resized)
                #io.imshow(im_resized,cmap='gray')
                #io.show()
                #im_resized = 255 - im_resized
                io.imsave(path_save + name.split('/')[-1],im_resized)
            else:
                print(name, ' Imagem vazia!')
        except:
            print(name)

N_THREADS = 60

path_read_images = ''
path_save_images = ''


name_images = glob.glob(path_read_images + '*.png')

n_records = len(name_images)

# launching subprocesses
for i in range(N_THREADS):
    threading.Thread(target=pre_processamento_transformar_imagens_em_quadradas_redimensionar_paralelo, args=(i, 
            (i*math.ceil(n_records/N_THREADS)), name_images[(i*math.ceil(n_records/N_THREADS)) : (math.ceil(n_records/N_THREADS) * (i + 1))],
            path_save_images,256,256)).start()

