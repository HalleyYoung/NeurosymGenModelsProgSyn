from skimage.io import imread, imsave
import os
import numpy as np
import pickle
from skimage.transform import resize
import torch
from skimage import img_as_ubyte, img_as_float
from prog2prog import NeuralNet


class ForTup():
	def __init__(self, i_n, j_n, i_offset, j_offset, i_mul, j_mul, i_size, j_size):
		self.i_n = i_n
		self.j_n = j_n
		self.i_offset = i_offset
		self.j_offset = j_offset
		self.i_mul = i_mul
		self.j_mul = j_mul
		self.i_size = i_size
		self.j_size = j_size
	def __str__(self):
		return "i_n: " + str(self.i_n) + " i_offset: " + str(self.i_offset) + " i_mul: " + str(self.i_mul) + " i_size: " + str(self.i_size) \
		+ " j_n: " + str(self.j_n) + " j_offset: " + str(self.j_offset) + " j_mul: " + str(self.j_mul) + " j_size: " + str(self.j_size)


def toMat(for_tup, img):
	x = np.zeros(8)
	x[0] = for_tup.i_offset/9.0
	x[1] = for_tup.j_offset/9.0
	x[2] = for_tup.i_n/9.0
	x[3] = for_tup.j_n/9.0
	x[4] = for_tup.i_size/5.0
	x[5] = for_tup.j_size/5.0
	x[6] = for_tup.i_mul/9.0
	x[7] = for_tup.j_mul/9.0
	return x


model = NeuralNet()
model.load_state_dict(torch.load("nn_epoch_550.pth"))
model.eval()


#progs_ = pickle.load(open("mat2mat.pcl"))
#progs = [np.array([progs_[i][k][0] for k in range(12)]) for i in range(9999)]
#comp_progs = [np.array([progs_[i][k][1] for k in range(12)]) for i in range(9999)]

imgs = []
progs = sorted(list(os.walk("test_third_prog"))[0][2])

for i_ in range(1000):
    try:
        img = img_as_float(resize(imread("facades/test/cmpb" + ("%04d" % i_) + ".png" + str(i), (256,256,3))))
        prog = progs[i_]
        new_prog = model(prog)[0].detach().numpy()[0,:,:]
        print("real " + str(new_prog[0,8:18]))
        new_img[:, 256/3:, :] = 0
        new_prog = comp_progs[i_]
        tot_good = 0
        for (index, for_tup_im) in enumerate(list(new_prog)):
                if np.isnan(for_tup_im).any() or (progs[i_] == np.zeros(18)).all():
                    continue
                tot_good += 1
                i_offset = int(for_tup_im[0]*9.01)
                j_offset = int(for_tup_im[1]*9.01)
                i_n = int(for_tup_im[2]*9.01)
                j_n = int(for_tup_im[3]*9.01)
                i_size = int(for_tup_im[4]*5.01)
                j_size = int(for_tup_im[5]*5.01)
                i_mul = max(1, int(for_tup_im[6]*9.01))
                j_mul =max(1, int(for_tup_im[7]*9.01))
                img_comp = torch.from_numpy(np.array([for_tup_im[8:18] for k in range(64)])).float()

                img = new_img[28*i_offset:(28 +  1)*i_offset, 28*j_offset:28*j_offset + 28, :]
                imgs.append(img)
                imsave("ims/im" + str(i_) + ".png",img)

                for i in range(i_n):
                    for j in range(j_n):
                        if j_offset + j*j_mul >= 3 and j_offset + j*j_mul < 9 and i_offset + i*i_mul < 9:
                            try:
                                new_img[28*(i*i_mul + i_offset):28*(i*i_mul + i_offset + i_size), 28*(j*j_mul + j_offset):28*(j*j_mul + j_offset + j_size), :3] = img_as_float(resize(img, (28*i_size, 28*j_size, 3)))
                                #print(resize(img, (17*i_size, 17*j_size, 3)))
                                #new_img[28*(i*i_mul + i_offset):28*(i*i_mul + i_offset + i_size), 28*(j*j_mul + j_offset):28*(j*j_mul + j_offset + j_size), 3] = np.full((28*i_size, 28*j_size), 255)
                                #print(new_img[28*(i*i_mul + i_offset):28*(i*i_mul + i_offset + i_size), 28*(j*j_mul + j_offset):28*(j*j_mul + j_offset + j_size),0])
                            except: pass
                                #print("error")
        #print(str(i_) + ": " + str(tot_good))

        imsave("facades_test/trainA/cmpb" + ("%04d" % i_) + ".png", new_img)
        insave("facades_test/trainB/cmpb" + ("%04d" % i_) + ".png", img)
    except:
        pass

"""
ims = list(imgs)
random.shuffle(ims)
imsave("imcomp.png", np.concatenate(ims[:64]))
"""
