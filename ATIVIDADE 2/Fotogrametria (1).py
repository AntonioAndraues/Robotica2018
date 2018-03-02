import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

# Para usar o vídeo
#cap = cv2.VideoCapture('hall_box_battery_mp2.mp4')

# As 3 próximas linhas são para usar a webcam
soma=0
z=0
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def distance_to_camera(knownWidth, focalLength, perWidth):
    # compute and return the distance from the maker to the camera
    return (knownWidth * focalLength) / perWidth
def identifica_cor(frame):
    '''
    Segmenta o maior objeto cuja cor é parecida com cor_h (HUE da cor, no espaço HSV).
    '''
    
    # No OpenCV, o canal H vai de 0 até 179, logo cores similares ao 
    # vermelho puro (H=0) estão entre H=-8 e H=8. 
    # Veja se este intervalo de cores está bom
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


#     cor_menor = np.array([100, 30, 50])
#     cor_maior = np.array([200, 200, 200])
    cor_menor = np.array([4, 50, 120])
    cor_maior = np.array([8, 255, 255])
    segmentado_cor = cv2.inRange(frame_hsv, cor_menor, cor_maior)
    kernel = np.ones((5,5),np.uint8)
    # segmentado_cor=cv2.morphologyEx(segmentado_cor, cv2.MORPH_OPEN, kernel)
    
    
    # x,y,w,h = cv2.boundingRect(cnt)
    # 2 cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    segmentado_cor=cv2.erode(segmentado_cor,kernel,iterations = 1)
    
    segmentado_cor = cv2.dilate(segmentado_cor,kernel,iterations = 1)
    segmentado_cor = cv2.morphologyEx(segmentado_cor, cv2.MORPH_OPEN, kernel)
    # Será possível limpar a imagem segmentado_cor? 
    # Pesquise: https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html


    # Encontramos os contornos na máscara e selecionamos o de maior área
    img_out, contornos, arvore = cv2.findContours(segmentado_cor.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    maior_contorno = None
    maior_contorno_area = 0
    global z
    global soma

    cv2.drawContours(frame, contornos, -1, (100, 100, 100), 5)


    for cnt in contornos:
        area = cv2.contourArea(cnt)


        if area > maior_contorno_area:
            maior_contorno = cnt
            maior_contorno_area = area
    #print(maior_contorno[0][0][1])
    
    auxmenor_i=0
    auxmaior_i=100000

    aux_maior=0
    aux_menor=1000000

    if not maior_contorno is None:
        for i in range(len(maior_contorno)):
            if maior_contorno[i][0][1]>aux_maior:
                aux_maior=maior_contorno[i][0][1]
                auxmaior_i=i

            if maior_contorno[i][0][1]<aux_menor:
                aux_menor=maior_contorno[i][0][1]
                auxmenor_i=i   
                


        
    # print("altura:{0}".format(aux_maior-aux_menor))
    alturabox=aux_maior-aux_menor
    distancia_da_camera=(108.5*217)/alturabox   
    TEXTOimput="distancia : {0}".format(distancia_da_camera)
    print("distancia : {0}".format(distancia_da_camera))
    # colorTEXT= blue
    cv2.putText(segmentado_cor,TEXTOimput,(20,450),cv2.FONT_ITALIC,1,(255,0,255),2)
    # Encontramos o centro do contorno fazendo a média de todos seus pontos.
    if not maior_contorno is None :
        cv2.drawContours(frame, [maior_contorno], -1, [0, 0, 255], 5)
        maior_contorno = np.reshape(maior_contorno, (maior_contorno.shape[0], 2))
        media = maior_contorno.mean(axis=0)
        media = media.astype(np.int32)
        cv2.circle(frame, tuple(media), 5, [0, 255, 0])
    else:
        media = (0, 0)

    cv2.imshow('', frame)
    cv2.imshow('imagem in_range', segmentado_cor)
    cv2.waitKey(1)

    centro = (frame.shape[0]//2, frame.shape[1]//2)

    return media, centro


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    img = frame.copy()

    media, centro = identifica_cor(img)

    #More drawing functions @ http://docs.opencv.org/2.4/modules/core/doc/drawing_functions.html

    # Display the resulting frame
    cv2.imshow('original',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
