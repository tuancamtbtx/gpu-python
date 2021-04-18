import cv2
import numpy as np
from numba import jit
from scipy.ndimage.filters import convolve, convolve1d

import argparse

def rgb2gray(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])

def rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(img, k)

@jit
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

def get_minimum_seam(img):
    # r, c = img.shape[:2]
    # energy_map = calc_energy(img)

    # M = energy_map.copy()
    # backtrack = np.zeros_like(M, dtype=np.int)

    # # TODO: 
    # return M, backtrack

    pass

def remove_seams(img, num_remove, rot=False):
    pass

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