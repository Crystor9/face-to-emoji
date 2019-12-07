import random
import colorsys
from PIL import Image

def josephChooser(hsv):
    if hsv[2] < .25:
        return 5
    elif hsv[1] < .25:
        return 6
    elif hsv[0] == 1.0:
        return 0
    return int(hsv[0] * 5)

new_size = 10000  # largest size of image that we resize to
small_size = 100  # smallest size of the image that we resize to

original = Image.open('joe.jpg')
emotions = [Image.open("josephs/" + str(i) + ".png") for i in range(7)]

joseph_size = int(new_size / small_size)
emotions = [joseph.resize((joseph_size, joseph_size)) for joseph in emotions]

new_im = original.resize((small_size, small_size), Image.LANCZOS)
hsvs = [colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255) for rgb in list(new_im.getdata())]

new_im = new_im.resize((new_size, new_size))
i = 0
for y in range(0, new_size, joseph_size):
    for x in range(0, new_size, joseph_size):
        joseph = emotions[josephChooser(hsvs[i])]
        new_im.paste(joseph, (x, y), joseph)
        i += 1

new_im.save('resized.png')
