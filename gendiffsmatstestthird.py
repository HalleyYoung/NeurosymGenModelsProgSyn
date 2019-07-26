import os
from skimage.io import imread, imsave, imshow
import copy
import random
#from skimage.measure import compare_mse, compare_ssim
import cv2
import numpy as np
#from skimage.transform import resize
import itertools
import torch
import pickle
from pyemd import emd_samples
#from skimage.color import rgb2gray, gray2rgb
#from skimage import img_as_ubyte
#from threading import Thread


def concat(xss):
	new = []
	for xs in xss:
		new.extend(xs)
	return new

orb = cv2.ORB_create(10000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def toMats():
    images = sorted(list(os.walk("facades/test"))[-1][2])
    for (index, image) in enumerate(images):
		print(index)
		img = imread("facades/test/" + image)
		img = img[:,:,:3]
		tot_height = img.shape[0]
		tot_width = img.shape[1]


		orb_val = orb.detectAndCompute(img, None)


		divisions = 15

		H = [int((tot_height+0.0)*i/divisions) for i in range(divisions)] + [tot_height]
		W = [int((tot_width+0.0)*i/divisions) for i in range(divisions)] + [tot_width]

		H.sort()
		W.sort()

		img_mat_loose = np.zeros((divisions,divisions,divisions,divisions))
		img_mat_tight = np.zeros((divisions,divisions,divisions,divisions))
		img_mat_looser = np.zeros((divisions, divisions, divisions, divisions))
		img_third = np.zeros((divisions, divisions, divisions, divisions))

		contrast = 64
		brightness = 127
		sifts = {}
		pix = {}
		for i in range(divisions):
			for j in range(divisions):
				pix[(i,j)] = np.concatenate(img[H[i]:H[i + 1], W[j]:W[j+1]])
				sift_indices = []
				for k in range(len(orb_val[0])):
					if orb_val[0][k].pt[0] >= i*tot_height/divisions and orb_val[0][k].pt[0] < (i+1)*tot_height/divisions  \
					and orb_val[0][k].pt[1] >= j*tot_width/divisions and orb_val[0][k].pt[1] < (j+1)*tot_width/divisions:
						sift_indices.append(k)
				

				sifts[(i,j)] = np.asarray([orb_val[1][k] for k in sift_indices], dtype="uint8")

		for i1 in range((divisions)):
			for j1 in range((divisions)):
				for i2 in range(i1, (divisions)):
					for j2 in range((divisions)):
						if not ((i1 == i2) and (j1 == j2)): 
							e = emd_samples(pix[(i1,j1)], pix[(i2,j2)])
							a = sifts[(i1,j1)]
							b = sifts[(i2,j2)]
							a_pix = pix[(i1,j1)]
							b_pix = pix[(i2,j2)]
							if not (len(a) == 0 or len(b) == 0):
								matches = bf.knnMatch(a, b, k=2)
								if len(matches) > 0 and len(matches[0]) == 2:
									good = []
									for m,n in matches:
									    if m.distance < 50.0:
									        good.append([m])
									if len(good) > 5 and e < 25.0:
										img_mat_tight[i1,j1,i2,j2] = 1
									elif len(good) > 5 and e < 35.0:
										img_mat_loose[i1,j1,i2,j2] = 1 
										img_mat_looser[i1,j1,i2,j2] = 1
						e = emd_samples(pix[(i1,j1)], pix[(i2,j2)])
						if e < 15:
							img_mat_tight[i1,j1,i2,j2] = 1
							img_mat_loose[i1,j1,i2,j2] = 1
							img_mat_looser[i1,j1,i2,j2] = 1
						elif e < 25:
							img_mat_loose[i1,j1,i2,j2] = 1
							img_mat_looser[i1,j1,i2,j2] = 1
						elif e < 30:
							img_mat_loose[i1,j1,i2,j2] = 1


						if j1 < 5 and j2 < 5:
							img_third[i1,j1,i2,j2] = img_mat_loose[i1,j1,i2,j2]
						else:
							pass #print((j1,j2))

						
		#np.save("mat_good.npy", img_mat_loose)
		np.save("buildings_mat_test_third/mat-" + image.split(".")[0] + ".npy", img_third)
        #np.save("buildings_mat_test_full/mat-" + image.split(".")[0] + ".npy", img_mat_loose)			
		print("saved")


toMats()
