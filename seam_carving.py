import numpy as np
from numba import jit
import cv2
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
def get_minimum_seam(img):
	r, c, _ = img.shape
	grad_mag_map = calc_energy(img)

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

	arg_parse.add_argument("-mode", help="Type of running seam: cpu or gpu", default=0)

	arg_parse.add_argument("-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
	arg_parse.add_argument("-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)

	arg_parse.add_argument("-in", help="Path to image", required=True)
	arg_parse.add_argument("-out", help="Output file name", required=True)


	args = vars(arg_parse.parse_args())

	in_img_path, out_img_path, dx, dy = args["in"], args["out"], args["dx"], args["dy"]

	if args["mode"] == "gpu":
		# TODO: later
		pass
	
	elif args["mode"] == "cpu":
		# TODO: resize input images base on dx and dy seam number

		in_img = cv2.imread(in_img_path)
		# visualize_image(in_img)
		out_img = main(in_img, dx, dy)

		cv2.imwrite(out_img_path, out_img)
		# visualize_image(out_img)
		pass
