import os
import random
import shutil

os.chdir("facades")
a = list(os.walk("./"))[0][2]

a = [i for i in a if i.split(".")[-1] in ["jpg","png"]]
random.shuffle(a)

total = len(a)

train = a[:(total/10)]
test = a[(total/10):]


os.mkdir("train")
os.mkdir("test")

for val in train:
	shutil.copy(val, "train/" + val)

for val in test:
	shutil.copy(val, "test/" + val)