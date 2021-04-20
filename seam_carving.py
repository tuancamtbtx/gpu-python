import cv2
import numpy as np
from numba import jit, njit
from scipy.ndimage.filters import convolve, convolve1d

import argparse

@njit
def rgb2gray(img):
    img = img.astype(np.uint8)
    h, w = img.shape[:2]
    gray_img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            gray_img[i][j] = img[i][j][0]*0.2989 + img[i][j][1]*0.5870 + img[i][j][2]*0.1140

    return gray_img

def rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(img, k)

@njit
def convolve2d(grayscale, filter_dx, filter_dy):
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
            energy_map[y, x] = abs(gx) + abs(gy)
            
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
    energy_map = convolve2d(grayscale, filter_dx, filter_dy)

    return energy_map

@njit
def get_minimum_seam(img):
    h, w = img.shape[:2]

    M = calc_energy(img) # matrix to store minimum energy value seen upon pixel
    backtrack = np.zeros_like(M, dtype=np.uint8)

    # TODO:
    for i in range(1, h):
        for j in range(0, w):
            # Handle the left edge of the image
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + 1
                min_energy = M[i-1, idx+j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            
            M[i, j] +=  min_energy

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
    
    # seam_idx.reverse()
    seam_idx = [seam_idx[-i - 1] for i in range(len(seam_idx))]
    return np.array(seam_idx), bool_mask

@jit
def remove_seam(img, bool_mask):
    h, w = img.shape[:2]
    # print(bool_mask.shape)
    # mask3c = np.stack([bool_mask] * 3, axis=2)
    # print(mask3c.shape)
    # mask3c.append([bool_mask])
    mask3c = np.empty((h, w, 3),dtype=np.bool8)
    for i in range(h):
        for j in range(w):
            for k in range(3):
                mask3c[i][j][k] = bool_mask[i][j]
    # print(mask3c.shape)
    # print(type(mask3c[0][0][1]))

    return img[mask3c].reshape((h, w-1, 3))

def remove_seams(img, num_remove, rot=False):
    for _ in range(num_remove):
        _, bool_mask = get_minimum_seam(img)
        img = remove_seam(img, bool_mask)

    return img

def insert_seams(img, num_insert, rot=False):
    pass

def seam_carving(img, dx, dy):
    # img = img.astype(np.float64)
    # h, w = img.shape[:2]
    # assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w

    # output = img

    # if dx < 0:
    #     output = remove_seams(output, -dx)

    # elif dx > 0:
    #     output = insert_seams(output, dx)

    # if dy < 0:
    #     output = rotate_image(output, True)
    #     output = remove_seams(output, -dy, rot=True)
    #     output = rotate_image(output, False)

    # elif dy > 0:
    #     output = rotate_image(output, True)
    #     output = insert_seams(output, dy, rot=True)
    #     output = rotate_image(output, False)

    # return output
    pass


if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("-mode", help="Type of running seam: cpu or gpu", type=str, default=0)

    arg_parse.add_argument("-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
    arg_parse.add_argument("-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)

    arg_parse.add_argument("-in", help="Path to image", required=True)
    arg_parse.add_argument("-out", help="Output file name", required=True)

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