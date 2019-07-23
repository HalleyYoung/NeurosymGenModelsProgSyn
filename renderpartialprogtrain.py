from skimage.io import imread, imsave
import os
import numpy as np
from prog2prog import NeuralNet
import pickle
from skimage.transform import resize
import torch
from skimage import img_as_ubyte, img_as_float

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
progs = [pickle.load("facades-approx-full-train-prog" + "/" + "cmpb" + ("%04d" % z_) + "-prog.pcl") for z_ in range(1000)]

progs = [np.array([progs_[i][k][0] for k in range(12)]) for i in range(9999)]
comp_progs = [np.array([progs_[i][k][1] for k in range(12)]) for i in range(9999)]
imgs = []

def toImageArr(for_tup_arr, fname, img):
	img2 = np.zeros((img.shape[0], img.shape[1], 3))

	tot_height = img.shape[0]
	tot_width = img.shape[1]

	H = [int((tot_height+0.0)*i/divisions) for i in range(divisions)] + [tot_height] 
	W = [int((tot_width+0.0)*i/divisions) for i in range(divisions)] + [tot_width]

	xs = []
	ys = []
	for for_tup in for_tup_arr:
		all_copies = []
		#img_index += 1
		#img1 = copy.copy(img[H[for_tup.i_offset]:H[for_tup.i_offset+1], W[for_tup.j_offset]:W[for_tup.j_offset + 1]])
		#imsave("testout" + str(img_index) + ".png", img1)
		
		for i in range(for_tup.i_n):
			for j in range(for_tup.j_n):
				try:
					y1 = H[i*for_tup.i_mul + for_tup.i_offset]
					y2 = H[i*for_tup.i_mul + for_tup.i_offset + 1]
					x1 = W[j*for_tup.j_mul + for_tup.j_offset]
					x2 = W[j*for_tup.j_mul + for_tup.j_offset + 1]
					if y2 > y1 and x2 > x1: #and img[y1:y2, x1:x2]:
						all_copies.append((i*for_tup.i_mul + for_tup.i_offset, j*for_tup.j_mul + for_tup.j_offset, copy.copy(img[y1:y2, x1:x2])))
				except: pass
		if len(all_copies) > 0:
			#imsave("testcopies" + str(random.randint(0,100)) + ".png", np.concatenate(all_copies))
			#print(all_copies)
			all_copies = [(i, sum(map(lambda k: emd_samples(i[2], k[2], bins = 20), all_copies))) for i in all_copies]
			all_copies.sort(key = lambda i: i[1])
			best = all_copies[0][0]
			img1 = copy.copy(img[H[best[0]]:H[min(len(H) - 1, best[0] + for_tup.i_size)], W[best[1]]:W[min(len(W) - 1, best[1] + for_tup.j_size)]])
                        #imsave(fname.split("/")[0] + "part-" + str(random.randint(0,100)) + "-" + fname.split("/")[1], img1)
			sub_height = img1.shape[0]
			sub_width = img1.shape[1]
                        
			for i in range(for_tup.i_n):
				for j in range(for_tup.j_n):
						try:
							img3 = copy.copy(img2)
							y = H[i*for_tup.i_mul + for_tup.i_offset]
							x = W[j*for_tup.j_mul + for_tup.j_offset]
							#print(str(y) + ", " + str(y+sub_height) + ", " + str(x) + ", " + str(x+sub_width))
							img2[y:y+sub_height, x:x+sub_width, :] = copy.copy(img1)
							#print("isequal " + str(np.array_equal(img2, img3)))
							#print(img1)
							#print("iszeros " + str(np.array_equal(img1, np.zeros(img1.shape))))
							#print("finished copying img1 to img2")
						except: pass
		imsave("facade-prog-only-train/" + fname, img2)
		img2[:,:85,:] = img[:,:85,:]
		imsave("facade-prog-third-train/" + fname, img2)
		img2[:,85:,:] = np.zeros((256,256-85,3))
		imsave("facade-third-only-train/" + fname, img2)


for i_ in range(1000):
    new_img = img_as_float(resize(imread("facades/train/cmpb" + ("%04d" % i_) + ".png" + str(i), (256,256,3))))
    prog = torch.from_numpy(np.array([progs[i_] for k in range(64)])).float()
    #new_img[:, 256/3:, :] = 0
    toImageArr(prog, "cmpb" + "%04d" % i_, new_img)

"""
ims = list(imgs)
random.shuffle(ims)
imsave("imcomp.png", np.concatenate(ims[:64]))
"""
