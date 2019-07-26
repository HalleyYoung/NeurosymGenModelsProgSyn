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

for (val, i) in enumerate(train):
	shutil.copy(val, "train/cmpb" + ("%04d" % i) + val.split("-")[-1])

for (val, i) in enumerate(test):
	shutil.copy(val, "test/cmpb" + ("%04d" % i) + val.split("-")[-1])
