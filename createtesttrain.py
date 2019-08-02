import os
import random
from skimage.io import imread, imsave
from skimage.transform import resize

os.chdir("facades")
a = list(os.walk("./"))[0][2]

a = [i for i in a if i.split(".")[-1] in ["jpg","png"]]
random.shuffle(a)

total = len(a)

train = a[:(total/10)]
test = a[(total/10):]

if not os.path.isdir("train"):
    os.mkdir("train")
if not os.path.isdir("test"):
    os.mkdir("test")

for (i, val) in enumerate(train):
    a = resize(imread(val), (256,256,3))
    imsave("train/cmpb" + ("%04d" % i) + ".png", a)

for (i, val) in enumerate(test):
    a = resize(imread(val), (256,256,3))
    imsave("test/cmpb" + ("%04d" % i) + ".png", a)
