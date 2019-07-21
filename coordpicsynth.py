from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from skimage.io import imsave
from skimage.transform import resize
import pickle
from pyemd import emd_samples
import numpy as np
from convtilessynth2 import VAE
from conv_sim import ImSim
from skimage import img_as_float
from skimage.color import rgb2gray
import random

vaemodel = VAE(7)
vaemodel.load_state_dict(torch.load("conv_tiles_synth7/vae_epoch_150.pth"))
vaemodel.eval()

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
	x = np.zeros(15)
	x[0] = for_tup.i_offset/9.0
	x[1] = for_tup.j_offset/9.0
	x[2] = for_tup.i_n/9.0
	x[3] = for_tup.j_n/9.0
	x[4] = for_tup.i_size/9.0
	x[5] = for_tup.j_size/9.0
	x[6] = for_tup.i_mul/9.0
	x[7] = for_tup.j_mul/9.0
	(a, b) = vaemodel.encode(torch.from_numpy(np.array([np.transpose(img_as_float(img)) for i in range(64)])).float())
	x[8:15] = vaemodel.reparametrize(a, b).float().detach().numpy()[0,:]
	img_comp = torch.from_numpy(np.array([x[8:18] for k in range(64)])).float()
	#imsave("compims/testsim" + str(random.randint(0,100)) + ".png", np.concatenate([np.transpose(vaemodel.decode(img_comp).detach().numpy()[0,:,:,:]), img]))
	return x

sim_model = ImSim()
sim_model.load_state_dict(torch.load("simmodel-200.ckpt"))

close = 0
far = 0
data_points = []
for i in range(5000):
	data_point = []
	print(i)
	#a = pickle.load(open("synth-approx-third-train-synth/" + str(i) + "-prog.pcl"))
	#a.sort(key = lambda k: (k[0].i_offset, k[0].j_offset))
	b = pickle.load(open("synth-approx-full-train-synth/" + str(i) + "-prog.pcl"))
	b.sort(key = lambda k: (k[0].i_offset, k[0].j_offset))

	if len(b) > 0:
		for (b_for_tup, (b_img, _)) in b:
			im1 = rgb2gray(a_img)
			im2 = rgb2gray(b_img)
			im1_empt = (im1 != 0).all() or (im1 == 0).all()
			im2_empt = (im2 != 0).all() or (im2 == 0).all()
			a_torch = torch.from_numpy(np.array([(im1 != 0) for q in range(64)]).astype(np.float)).view((64,1,16,16)).float()
			b_torch = torch.from_numpy(np.array([(im2 != 0) for q in range(64)]).astype(np.float)).view((64,1,16,16)).float()
			emd_dist = emd_samples(np.concatenate(a_img), np.concatenate(b_img))
			sim = sim_model(a_torch, b_torch)
			if (np.isclose(a_img, b_img, rtol=0.10).all() or (im1_empt and im2_empt) or (sim[0][1] < sim[1][1] and im1_empt == im2_empt)) and emd_dist < 0.15:
				mat_a = toMat(a_for_tup, a_img)
				mat_b = toMat(b_for_tup, b_img)
				#imsave("ims/im" + str(i) + ".png", np.concatenate([a_img,b_img]))
			data_point.append(mat_b)
	data_point.sort(key = lambda k: (k[0][0], k[0][1]))
	print("len: " + str(len(data_point)))
	if len(data_point) < 12:
		new_points = [(np.zeros(18), toMat(b[i][0], b[i][1][0])) for i in range(12-len(data_point))]
	data_points.append(data_point + new_points)

pickle.dump(data_points, open("matsynth.pcl", "w+"))
