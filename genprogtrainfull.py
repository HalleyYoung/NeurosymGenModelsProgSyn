import pickle
import random
import copy
import numpy as np
import itertools
from skimage.io import imread, imsave
from pyemd import emd_samples
from skimage.transform import resize
import os
from skimage import img_as_float
from skimage import img_as_ubyte

img_size = 256
tot_height = img_size
tot_width = img_size
divisions = 15

def concat(xss):
	new = []
	for xs in xss:
		new.extend(xs)
	return new


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
		


def matFromForTup(for_tup, img_mat):
	adj = np.zeros(img_mat.shape)
	i_s = [i*for_tup.i_mul + for_tup.i_offset for i in range(for_tup.i_n)]
	i_s = filter(lambda i: i + for_tup.i_size < img_mat.shape[0], i_s)
	j_s = [j*for_tup.j_mul + for_tup.j_offset for j in range(for_tup.j_n)]
	j_s = filter(lambda j: j + for_tup.j_size < img_mat.shape[1], j_s)
	i_s_pairs = [k for k in list(itertools.product(i_s, i_s)) if k[0] <= k[1]]
	j_s_pairs = [k for k in list(itertools.product(j_s, j_s)) if k[0] <= k[1]]
	i_s_j_s = list(itertools.product(i_s_pairs, j_s_pairs))
	for ((i1, i2), (j1, j2)) in i_s_j_s:
		i_size = for_tup.i_size
		j_size = for_tup.j_size
		for i_ in range(i_size):
			for j_ in range(j_size):
				adj[(i1+i_), (j1+j_), (i2+i_), (j2 + j_)] = 1
				adj[(i2+i_), (j2 + j_), (i1 + i_), (j1 + j_)] = 1
	return adj


everything_but_self = np.ones((divisions, divisions, divisions, divisions))
for i in range(divisions):
	for j in range(divisions):
		everything_but_self[i,j,i,j] = 0

def diffMats(img_mat, loop_mat):

	unique, counts = np.unique(img_mat - loop_mat, return_counts = True)
	counts = dict(zip(unique, counts))
	#should_be_connected_diff = 0.8*counts[1.0] if 1.0 in counts else 0
	correct_connected_diff = np.count_nonzero(np.logical_and(everything_but_self, np.logical_and(img_mat == loop_mat, img_mat == 1)))
	shouldnt_be_connected_diff = counts[-1.0] if -1.0 in counts else 0
	#print("should_be_connected_diff: " + str(should_be_connected_diff) + " shouldnt_be_connected_diff: " + str(shouldnt_be_connected_diff))
	return (2*correct_connected_diff - shouldnt_be_connected_diff, correct_connected_diff, shouldnt_be_connected_diff)

def toProg(for_tup_arr, fname, img):
	tups = []
        H = [int((tot_height+0.0)*i/divisions) for i in range(divisions)] + [tot_height]
        W = [int((tot_width+0.0)*i/divisions) for i in range(divisions)] + [tot_width]


	for for_tup in for_tup_arr:
                #print("in for_tup")
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
						all_copies.append(copy.copy(img[y1:y2, x1:x2]))
				except: pass
                #print(all_copies)
		if len(all_copies) > 0:
			#imsave("testcopies" + str(random.randint(0,100)) + ".png", np.concatenate(all_copies))
			all_copies = [(i, sum(map(lambda k: emd_samples(np.concatenate(i), np.concatenate(k), bins = 100), all_copies))) for i in all_copies]
			all_copies.sort(key = lambda i: i[1])
			best = all_copies[0]
			tups.append((for_tup, best))
		else:
			continue	
	f = open(fname, "w+")
	pickle.dump(tups, f)
	f.close()



def getCoverage(f):
	i_n = len([i for i in range(f.i_n) if f.i_mul*i + f.i_offset + f.i_size - 1 < divisions])
	j_n = len([j for j in range(f.j_n) if f.j_mul*j + f.j_offset + f.j_size - 1 < divisions])
	return i_n*j_n

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


buildings = [img_as_float(imread("facades/train/" + i)) for i in sorted(list(os.walk("facades/train"))[-1][2])]
image_mats = [np.load("buildings_mat_train_full/" + i) for i in sorted(list(os.walk("buildings_mat_train_full"))[-1][2])]

for z_ in range(len(image_mats)):
	print(z_)
	#img_mat = np.load("buildings_mat_train_full/" + str(image_mats[z_]))
	img_mat = image_mats[z_]
	img = buildings[z_][:,:,:3]
	good_for_tups = []
	for i_offset in range(15):
		print(i_offset)
		#print("i_offset: " + str(i_offset))
		for j_offset in range(15):
			best_in_off = []
			seen = []
			#print("j_offset: " + str(j_offset))
			for i_mul in range(2,7):
				for j_mul in range(2,7):
					ns = [(1,2), (2,1)]
					for (i_n,j_n) in ns:
						f = ForTup(i_n = i_n, i_offset = i_offset, i_mul = i_mul, i_size = 1, j_size = 1, j_n = j_n, j_offset = j_offset, j_mul = j_mul)
						if str(f) not in seen:
							seen.append(str(f))
							score = diffMats(img_mat, matFromForTup(f, img_mat))[0]
							if score > 0:
								while True:
									#print("in " + str(f) + " score: " + str(score))
									new_fs = [copy.copy(f), copy.copy(f)]
									new_fs[0].i_n += 1
									new_fs[1].j_n += 1
									if f.i_size < 5 and f.i_n > 1 and f.i_size < f.i_mul - 1:
										f_ = copy.copy(f)
										f_.i_size += 1
										new_fs.append(f_)
									if f.j_size < 5 and f.j_n > 1 and f.j_size < f.j_mul - 1:
										f_ = copy.copy(f)
										f_.j_size += 1
										new_fs.append(f_)				
									new_fs = [k for k in new_fs if str(k) not in seen]
									if len(new_fs) == 0:
										break
									seen.extend(map(str, new_fs))
									mat_scores = [diffMats(img_mat, matFromForTup(k, img_mat))[0] for k in new_fs]
									if max(mat_scores) <= score:
										break
									else:
										f = max([(new_fs[k], mat_scores[k]) for k in range(len(new_fs))], key = lambda q: q[1])[0]
										score = max(mat_scores)
									best_in_off.append((f, score))
									#print(str(f))
			if len(best_in_off) > 0:
				good_for_tups.append(max(best_in_off, key = lambda k: k[1]))



	good_for_tups.sort(key = lambda i: getCoverage(i[0]), reverse=True)
	good_for_tups = good_for_tups[:30]

	good_for_tups = [i[0] for i in good_for_tups]
	full_set = []
	inclusion = set([])
	for for_tup in good_for_tups:
		i_s = [x for x in concat([range(i*for_tup.i_mul+for_tup.i_offset,(i*for_tup.i_mul + for_tup.i_offset+ for_tup.i_size)) for i in range(for_tup.i_n)]) if x < divisions]
		j_s = [x for x in concat([range(j*for_tup.j_mul+for_tup.j_offset,(j*for_tup.j_mul + for_tup.j_offset+ for_tup.i_size))for j in range(for_tup.j_n)]) if x < divisions]
		pairs = list(itertools.product(i_s, j_s))
		#print "len pairs: " + str(len(pairs)) + " getCoverage " + str(getCoverage(for_tup))
		if len([x for x in pairs if x not in inclusion]) > 1:
			#if len([x for x in pairs if x in inclusion]) < 20:
				for pair in pairs:
					inclusion.add(pair)
				full_set.append(for_tup)
	print "full-set: " + str(len(full_set))

	full_set = full_set[:10]

	full_set.reverse()
	toImageArr(full_set, "cmpb" + ("%04d" % z_) + ".png", img)
	#toMaskArr(full_set, "approx-" + str(img_size) + "-" + str(q__) + "/" + str(z_) + "--approx-mask.png", img)
	#toPngWithAlpha(q__, z_)
	toProg(full_set, "facades-approx-full-train-prog" + "/" + "cmpb" + ("%04d" % z_) + "-prog.pcl", img)

