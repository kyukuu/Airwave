# Programa modificado por Diego Ojeda a partir del codigo de Michael D'Argenio
# Podeís encontrar el código original aquí: https://www.hackster.io/mjdargen/easy-object-detection-with-teachable-machine-python-d4063b

import numpy as np
import cv2
import tensorflow as myTensor
import math
import os
import serial
import time

tf = myTensor.keras

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

arduino = serial.Serial(port='COM13', baudrate=9600, timeout=.1)#definimos nuestro arduino

def main():

    # read .txt file to get labels
    labels_path = f"{DIR_PATH}/labels.txt"
    # open input file label.txt
    labelsfile = open(labels_path, 'r')

    # initialize classes and read in lines until there are no more
    classes = []
    line = labelsfile.readline()
    while line:
        # retrieve just class name and append to classes
        classes.append(line.split(' ', 1)[1].rstrip())
        line = labelsfile.readline()
    # close label file
    labelsfile.close()

    # load the teachable machine model
    model_path = f"{DIR_PATH}/keras_model.h5"
    model = tf.models.load_model(model_path, compile=False)

    # initialize webcam video object
    cap = cv2.VideoCapture(0)

    # width & height of webcam video in pixels -> adjust to your size
    # adjust values if you see black bars on the sides of capture window
    frameWidth = 1280
    frameHeight = 720

    # set width and height in pixels
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frameWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frameHeight)
    # enable auto gain
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # keeps program running forever until ctrl+c or window is closed
    while True:
        # time.sleep(2)
        # disable scientific notation for clarity
        np.set_printoptions(suppress=True)

        # Create the array of the right shape to feed into the keras model.
        # We are inputting 1x 224x224 pixel RGB image.
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        # capture image
        check, frame = cap.read()
        # mirror image - mirrored by default in Teachable Machine
        # depending upon your computer/webcam, you may have to flip the video
        frame = cv2.flip(frame, 1)

        # crop to square for use with TM model
        #aquí lo que hacemos es añadirle margenes a la imagen para que sea cuadrada y luego así la podemos enviar a la teachable machine con un tamaño de 224x224
        margin = int(((frameWidth-frameHeight)/2))
        square_frame = frame[0:frameHeight, margin:margin + frameHeight]
        # resize to 224x224 for use with TM model
        resized_img = cv2.resize(square_frame, (224, 224))
        # convert image color to go to model
        #transforma la imagen de tipo BGR a RGB para que la teachable machine lo entienda
        model_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # turn the image into a numpy array
        image_array = np.asarray(model_img)
        # normalize the image para que los pixeles esten en un rango entre -1 y 1
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        # load the image into the array
        data[0] = normalized_image_array

        # run the prediction
        predictions = model.predict(data)

        # confidence threshold is 90%.
        conf_threshold = 90
        confidence = []
        conf_label = ""
        threshold_class = ""
        # create blach border at bottom for labels
        per_line = 2  # number of classes per line of text
        bordered_frame = cv2.copyMakeBorder(
            square_frame,
            top=0,
            bottom=30 + 15*math.ceil(len(classes)/per_line),
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        # for each one of the classes
        for i in range(0, len(classes)):
            # scale prediction confidence to % and apppend to 1-D list
            confidence.append(int(predictions[0][i]*100))
            # put text per line based on number of classes per line
            if (i != 0 and not i % per_line):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil(i/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # append classes and confidences to text for label
            conf_label += classes[i] + ": " + str(confidence[i]) + "%; "
            # prints last line
            if (i == (len(classes)-1)):
                cv2.putText(
                    img=bordered_frame,
                    text=conf_label,
                    org=(int(0), int(frameHeight+25+15*math.ceil((i+1)/per_line))),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255, 255, 255)
                )
                conf_label = ""
            # if above confidence threshold, send to queue
            if confidence[i] > conf_threshold:
                num=0
                #aqui cambiar yo y nada por el nombre de vuestras clases
                if str(classes[i])=="No Gesture":
                    num=1
                elif str(classes[i])=="Thumbs Up":
                    num=2
                elif str(classes[i]=="Thumbs Down"):
                    num=3
                elif str(classes[i]=="Palms Spread"):
                    num=4
                elif str(classes[i]=="Ok"):
                    num=5
                elif str(classes[i]=="Rock"):
                    num=6
                arduino.write((str(num)+ '\n').encode())
                threshold_class = classes[i]
        # add label class above confidence threshold
        cv2.putText(
            img=bordered_frame,
            text=threshold_class,
            org=(int(0), int(frameHeight+20)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255)
        )

        # original video feed implementation
        cv2.imshow("Capturing", bordered_frame)
        cv2.waitKey(10)

if __name__ == '__main__':
    main()

