from PIL import Image, ImageDraw, ImageFilter
from matplotlib import pyplot as plt
import numpy as np
import face_recognition
import keras
from keras.models import load_model
import cv2
import sys


# Defaults
emotion_dict= {'Angry': 0, 'Sad': 5, 'Neutral': 4, 'Disgust': 1, 'Surprise': 6, 'Fear': 2, 'Happy': 3}
size = 10
place = (0,0)


# Default file if none are provided
default = 'multiple2.jpeg'
file = "face_and_emotion_detection2/test_images/" + default
output = './output/' + default

# Check for input (and output) arguments
arguments = len(sys.argv) - 1
print ("the script is called with %i arguments" % (arguments))

if arguments == 1:
    file = sys.argv[1]
elif arguments == 2:
    file = sys.argv[1]
    output = sys.argv[2]


# Load the jpg file into a numpy array, find faces with HOG model
# (not as good as CNN model)
image = face_recognition.load_image_file("./" + file)
face_locations = face_recognition.face_locations(image)


print("I found {} face(s) in this photograph.\n\n".format(len(face_locations)))

copyIm = Image.open('./' + file).copy()

# Find all the faces & paste Joseph over them
for face_location in face_locations:
    
    # Print the location of each face in this image
    top, right, bottom, left = face_location
    size = int((bottom - top) * 1.55)
    place = (int(left - .1 * size), int(top - .25 * size))

    # Access actual face
    face_image = image[top:bottom, left:right]
    face_image = cv2.resize(image, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    model = load_model("./face_and_emotion_detection2/emotion_detector_models/model_v6_23.hdf5")

    # find emotion of current face
    predicted_class = np.argmax(model.predict(face_image))
    label_map = dict((v,k) for k,v in emotion_dict.items())
    # predicted_label = label_map[predicted_class]
    # print(predicted_label)

    # Get the correct Joseph picture
    im1 = Image.open('./face_and_emotion_detection2/graphics/anger.png')

    if predicted_class == 1:
        im1 =Image.open('./face_and_emotion_detection2/graphics/disgust.png')
    elif predicted_class == 2:
        im1 =Image.open('./face_and_emotion_detection2/graphics/fear.png')
    elif predicted_class == 3:
        im1 =Image.open('./face_and_emotion_detection2/graphics/happy.png')
    if predicted_class == 4:
        im1 =Image.open('./face_and_emotion_detection2/graphics/neutral.png')
    elif predicted_class == 5:
        im1 =Image.open('./face_and_emotion_detection2/graphics/sad.png')
    elif predicted_class == 6:
        im1 =Image.open('./face_and_emotion_detection2/graphics/surprise.png')

    # paste face
    fml = im1.resize((size, size))
    copyIm.paste(fml, place, fml)

# SAVE :D
copyIm.save(output)
copyIm.show()
