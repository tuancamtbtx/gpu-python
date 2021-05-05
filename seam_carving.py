import cv2
import numpy as np
from numba import jit
import cv2
from numba import jit, njit
from scipy.ndimage.filters import convolve, convolve1d
from scipy import ndimage as ndi

import argparse
from scipy.ndimage.filters import convolve


FILTER_DX = np.array([
		[1.0, 2.0, 1.0],
		[0.0, 0.0, 0.0],
		[-1.0, -2.0, -1.0],
	])

FILTER_DY = np.array([
		[1.0, 0.0, -1.0],
		[2.0, 0.0, -2.0],
		[1.0, 0.0, -1.0],
	])

def rotate_image(img, clockwise=1):
	return np.rot90(img, clockwise)    


def visualize_image(img, boolmask=None, rotate=False):
	visuallize = img.astype(np.uint8)
	if boolmask is not None:
		visuallize[np.where(boolmask == False)] = np.array([255, 200, 200]) # BGR

	cv2.imshow("visualization", visuallize)
	cv2.waitKey(1)
	return visuallize


@jit(forceobj=True)
def calc_energy(img, filter_dx=FILTER_DX, filter_dy=FILTER_DY):
	"""
	Simple gradient magnitude energy map.
	"""
	# This converts it from a 2D filter to a 3D filter, replicating the same
	# filter for each channel: R, G, B
	filter_dxs = np.stack([filter_dx] * 3, axis=2)
	filter_dys = np.stack([filter_dy] * 3, axis=2)

	convolved = np.absolute(convolve(img, filter_dxs)) + np.absolute(convolve(img, filter_dys))

	# We sum the energies in the red, green, and blue channels
	grad_mag_map = convolved.sum(axis=2)

	return grad_mag_map

@jit(forceobj=True)
def forward_energy(im):
    """
    Forward energy algorithm as described in "Improved Seam Carving for Video Retargeting"
    by Rubinstein, Shamir, Avidan.
    Vectorized code adapted from
    https://github.com/axu2/improved-seam-carving.
    """
    h, w = im.shape[:2]
    im = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))
    
    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)
    R = np.roll(im, -1, axis=1)
    
    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU
    
    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)
    
    # vis = visualize(energy)
    # cv2.imwrite("forward_energy_demo.jpg", vis)     
        
    return energy

@jit(forceobj=True)
def get_minimum_seam(img):
	r, c, _ = img.shape
	# grad_mag_map = calc_energy(img)
	grad_mag_map = forward_energy(img)

	energy_map = grad_mag_map.copy()
	backtrack_loc = np.zeros_like(energy_map, dtype=np.int32)

	for i in range(1, r):
		for j in range(0, c):
			# Handle the left edge of the image, to ensure we don't index -1
			if j == 0:
				idx = np.argmin(energy_map[i - 1, j:j + 2])
				backtrack_loc[i, j] = idx + j
				min_energy = energy_map[i - 1, idx + j]
			else:
				idx = np.argmin(energy_map[i - 1, j - 1:j + 2])
				backtrack_loc[i, j] = idx + j - 1
				min_energy = energy_map[i - 1, idx + j - 1]

			energy_map[i, j] += min_energy

	# backtrack to find path
	seam_idx = []

	boolmask = np.ones((r, c), dtype=np.bool8)
	j = np.argmin(energy_map[-1])
	for i in range(r-1, -1, -1):
		boolmask[i, j] = False
		seam_idx.append(j)
		j = backtrack_loc[i, j]

	seam_idx.reverse()
	return np.array(seam_idx), boolmask

@jit(forceobj=True)
def remove_seam(img, boolmask):
	r, c, _ = img.shape
	boolmask_3c = np.stack([boolmask] * 3, axis=2)
	return img[boolmask_3c].reshape((r, c - 1, 3))


@jit(forceobj=True)
def add_seam(img, seam_idx):
	r, c, _ = img.shape
	out_img = np.zeros((r, c + 1, 3))
	for row in range(r):
		col = seam_idx[row]
		for channel in range(3):
			if col == 0:
				p = np.average(img[row, col: col + 2, channel])
				out_img[row, col, channel] = img[row, col, channel]
				out_img[row, col + 1, channel] = p
				out_img[row, col + 1:, channel] = img[row, col:, channel]
			else:
				p = np.average(img[row, col - 1: col + 1, channel])
				out_img[row, : col, channel] = img[row, : col, channel]
				out_img[row, col, channel] = p
				out_img[row, col + 1:, channel] = img[row, col:, channel]

	return out_img


@njit
def rgb2gray(img):
	img = img.astype(np.uint8)
	h, w = img.shape[:2]
	gray_img = np.zeros((h, w), dtype=np.float64)
	for i in range(h):
		for j in range(w):
			gray_img[i][j] = img[i][j][0]*0.2989 + \
				img[i][j][1]*0.5870 + img[i][j][2]*0.1140

	return gray_img


def rotate_image(img, clockwise):
	k = 1 if clockwise else 3
	return np.rot90(img, k)


@njit
def do_sobel(grayscale, filter_dx, filter_dy):
	# Add zero padding to the input image
	image_padded = np.zeros((grayscale.shape[0] + 2, grayscale.shape[1] + 2))
	image_padded[1:-1, 1:-1] = grayscale

	# energy_map = np.absolute(convolve(grayscale, filter_du)) + np.absolute(convolve(grayscale, filter_dv))
	energy_map = np.zeros_like(grayscale)
	for x in range(grayscale.shape[1]):
		for y in range(grayscale.shape[0]):
			# element-wise multiplication of the kernel and the image
			gx = (filter_dx * image_padded[y: y+3, x: x+3]).sum()
			gy = (filter_dy * image_padded[y: y+3, x: x+3]).sum()
			energy_map[y, x] = np.absolute(gx) + np.absolute(gy)

	return energy_map


@njit
def calc_energy(img):
	# convert rgb to grayscale
	grayscale = rgb2gray(img)

	filter_dx = np.array([
		[1.0, 2.0, 1.0],
		[0.0, 0.0, 0.0],
		[-1.0, -2.0, -1.0],
	])
	filter_dy = np.array([
		[1.0, 0.0, -1.0],
		[2.0, 0.0, -2.0],
		[1.0, 0.0, -1.0],
	])

	# sobel
	energy_map = do_sobel(grayscale, filter_dx, filter_dy)

	return energy_map

@njit
def get_minimum_seam(img):
	h, w = img.shape[:2]

	# matrix to store minimum energy value seen upon pixel
	M = calc_energy(img)
	backtrack = np.zeros_like(M, dtype=np.uint16)
	for r in range(1, h):
		for c in range(0, w):
			# Handle the left edge of the image
			if c == 0:
				idx = np.argmin(M[r-1, c:c + 2])
				backtrack[r, c] = idx + c
				min_energy = M[r-1, idx+c]
			# elif c == w-1:
			#     idx = np.argmin(M[r-1, c-1:c+1])
			#     backtrack[r, c] = idx + c - 1
			#     min_energy = M[r-1, idx + c - 1]
			else:
				idx = np.argmin(M[r - 1, c - 1:c + 2])
				backtrack[r, c] = idx + c - 1
				# print(backtrack[r, c])
				min_energy = M[r - 1, idx + c - 1]

			M[r, c] += min_energy

	# back tracking to find minimum cost path
	# column coordinations using for insert seams in later
	seam_idx = []
	# create a (h, w) matrix filled with the value True
	# and removing all pixels from the image which have False in later
	bool_mask = np.ones((h, w), dtype=np.bool8)
	# find the minum cost in bottom row
	j = np.argmin(M[-1])
	for i in range(h-1, -1, -1):
		bool_mask[i, j] = False
		seam_idx.append(j)
		j = backtrack[i, j]

	seam_idx = [seam_idx[-i - 1] for i in range(len(seam_idx))]
	return np.array(seam_idx), bool_mask


@jit
def remove_seam(img, bool_mask):
	h, w, _ = img.shape
	# print(bool_mask.shape)
	# mask3c = np.stack([bool_mask] * 3, axis=2)
	# print(mask3c.shape)
	# mask3c.append([bool_mask])
	mask3c = np.empty((h, w, 3), dtype=np.bool8)
	for i in range(h):
		for j in range(w):
			for ch in range(3):
				mask3c[i][j][ch] = bool_mask[i][j]

	return img[mask3c].reshape((h, w-1, 3))


def remove_seams(img, num_remove, rot=False):
	for _ in range(num_remove):
		_, bool_mask = get_minimum_seam(img)
		img = remove_seam(img, bool_mask)

	return img


@jit
def insert_seam(img, seam_idx):
	h, w = img.shape[:2]
	output = np.zeros((h, w+1, 3))
	# The inserted pixel values are derived from an
	# average of left and right neighbors.
	for r in range(h):
		c = seam_idx[r]
		for ch in range(3):  # chanel
			if c == 0:
				p = np.average(img[r, c:c+2, ch])
				output[r, c, ch] = img[r, c, ch]
				output[r, c+1, ch] = p
				output[r, c+1:, ch] = img[r, c:, ch]
			else:
				p = np.average(img[r, c - 1: c + 1, ch])
				output[r, :c, ch] = img[r, :c, ch]
				output[r, c, ch] = p
				output[r, c+1:, ch] = img[r, c:, ch]

	return output


def insert_seams(img, num_insert, rot=False):
	seam_idxs_record = []
	temp_img = img.copy()

	for _ in range(num_insert):
		seam_idx, bool_mask = get_minimum_seam(temp_img)

		seam_idxs_record.append(seam_idx)
		temp_img = remove_seam(temp_img, bool_mask)

	seam_idxs_record.reverse()

	for _ in range(num_insert):
		seam_idx = seam_idxs_record.pop()
		img = insert_seam(img, seam_idx)

		for remain_seam in seam_idxs_record:
			remain_seam[np.where(remain_seam >= seam_idx)] += 2

	return img


def seam_carving(img, dx, dy):
	img = img.astype(np.float64)
	h, w = img.shape[:2]
	assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

	output = img

	if dx < 0:
		output = remove_seams(output, -dx)

	elif dx > 0:
		output = insert_seams(output, dx)

	if dy < 0:
		output = rotate_image(output, True)
		output = remove_seams(output, -dy, rot=True)
		output = rotate_image(output, False)

	elif dy > 0:
		output = rotate_image(output, True)
		output = insert_seams(output, dy, rot=True)
		output = rotate_image(output, False)

	return output
def start_seams_removal(img, num_remove):

	new_img = img.copy()
	for _ in range(num_remove):
		seam_idx, boolmask = get_minimum_seam(new_img)
		new_img = remove_seam(new_img, boolmask)

	return new_img


def start_seams_insertion(img, num_add):
	seam_idxs_record = []
	temp_img = img.copy()

	for _ in range(num_add):
		seam_idx, bool_mask = get_minimum_seam(temp_img)

		seam_idxs_record.append(seam_idx)
		temp_img = remove_seam(temp_img, bool_mask)

	seam_idxs_record.reverse()

	for _ in range(num_add):
		seam_idx = seam_idxs_record.pop()
		img = add_seam(img, seam_idx)

		for remain_seam in seam_idxs_record:
			remain_seam[np.where(remain_seam >= seam_idx)] += 2


	return img





def main(img, dx, dy):
	img = img.astype(np.float64)
	r, c, _ = img.shape

	# prevent out of size of image
	assert c + dy > 0 and r + dx > 0 and dy <= c and dx <= r

	out_img = img

	if dx < 0:
		out_img = start_seams_removal(out_img, -dx)

	elif dx > 0:
		out_img = start_seams_insertion(out_img, dx)

	if dy < 0:
		out_img = rotate_image(out_img)
		out_img = start_seams_removal(out_img, -dy)
		out_img = rotate_image(out_img, clockwise=3)

	elif dy > 0:
		out_img = rotate_image(out_img)
		out_img = start_seams_insertion(out_img, dy)
		out_img = rotate_image(out_img, clockwise=3)

	return out_img

if __name__ == '__main__':
	arg_parse = argparse.ArgumentParser()

	arg_parse.add_argument("-in", help="Path to image", required=True)
	arg_parse.add_argument("-out", help="Output file name", required=True)

	arg_parse.add_argument(
		"-mode", help="Type of running seam: cpu or gpu", type=str, default=0)

	arg_parse.add_argument(
		"-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
	arg_parse.add_argument(
		"-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)

	args = vars(arg_parse.parse_args())

	IN_IMG, OUT_IMG = args["in"], args["out"]

	img = cv2.imread(IN_IMG)
	assert img is not None

	print(img.shape)

	if args["mode"] == "gpu":
		# TODO: later
		pass

	elif args["mode"] == "cpu":
		# TODO: resize input images base on dx and dy seam number
		dx, dy = args["dx"], args["dy"]
		assert dx is not None and dy is not None
		output = seam_carving(img, dx, dy)
		cv2.imwrite(OUT_IMG, output)