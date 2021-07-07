from PIL import Image
import cv2
import numpy as np
from numba import jit, njit, cuda
import math
import time
import hashlib

import argparse

# ignore numba warning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)


@njit
def rgb2gray(img):
    img = img
    h, w = img.shape[:2]
    gray_img = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            gray_img[i][j] = img[i][j][0]*0.2989 + \
                img[i][j][1]*0.5870 + img[i][j][2]*0.1140

    return gray_img


def rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.ascontiguousarray(np.rot90(img, k))


@njit
def calc_energy(gray_img):

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

    # Add zero padding to the input image
    image_padded = np.zeros((gray_img.shape[0] + 2, gray_img.shape[1] + 2))
    image_padded[1:-1, 1:-1] = gray_img

    # energy_map = np.absolute(convolve(grayscale, filter_du)) + np.absolute(convolve(grayscale, filter_dv))
    energy_map = np.zeros_like(gray_img)
    for x in range(gray_img.shape[1]):
        for y in range(gray_img.shape[0]):
            # element-wise multiplication of the kernel and the image
            gx = (filter_dx * image_padded[y: y+3, x: x+3]).sum()
            gy = (filter_dy * image_padded[y: y+3, x: x+3]).sum()
            energy_map[y, x] = np.absolute(gx) + np.absolute(gy)

    return energy_map


@njit
def forward_energy(gray_img):
    height, width = gray_img.shape[:2]

    energy = np.zeros((height, width))
    m = np.zeros((height, width))

    for r in range(1, height):
        for c in range(width):
            up = (r-1) % height
            left = (c-1) % width
            right = (c+1) % width

            mU = m[up, c]
            mL = m[up, left]
            mR = m[up, right]

            cU = np.abs(gray_img[r, right] - gray_img[r, left])
            cL = np.abs(gray_img[up, c] - gray_img[r, left]) + cU
            cR = np.abs(gray_img[up, c] - gray_img[r, right]) + cU

            cULR = np.array([cU, cL, cR])
            mULR = np.array([mU, mL, mR]) + cULR

            argmin = np.argmin(mULR)

            m[r, c] = mULR[argmin]
            energy[r, c] = cULR[argmin]

    return energy


@njit
def get_minimum_cost_table(energy):
    h, w = energy.shape[:2]
    backtrack = np.zeros_like(energy, dtype=np.uint16)
    for r in range(1, h):
        for c in range(0, w):
            left = max(0, c - 1)
            right = min(w - 1, c + 1)

            min_energy = energy[r-1, left]
            backtrack_col = left
            if energy[r-1, c] < min_energy:
                min_energy = energy[r-1, c]
                backtrack_col = c
            if energy[r-1, right] < min_energy:
                min_energy = energy[r-1, right]
                backtrack_col = right

            energy[r, c] += min_energy
            backtrack[r, c] = backtrack_col

    return energy, backtrack


@njit
def get_minimum_seam(min_costs, backtrack):
    h, w = min_costs.shape[:2]
    # back tracking to find minimum cost path
    seam_idx = []
    # create a (h, w) matrix filled with the value True
    # and removing all pixels from the image which have False in later
    bool_mask = np.ones((h, w), dtype=np.bool8)
    # find the minum cost in bottom row
    j = np.argmin(min_costs[-1])
    for i in range(h-1, -1, -1):
        bool_mask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx = [seam_idx[-i - 1] for i in range(len(seam_idx))]
    return np.array(seam_idx), bool_mask


@njit
def remove_seam(img, seam_idxs):
    height, width, chanel = img.shape

    output = np.empty((height, width-1, chanel), dtype=np.float64)
    for row in range(height):
        for col in range(width):
            if seam_idxs[row] <= col and col < width-1:
                for ch in range(chanel):
                    output[row][col][ch] = img[row][col + 1][ch]
            else:
                for ch in range(chanel):
                    output[row][col][ch] = img[row][col][ch]

    return output


def remove_seams(img, num_remove, test_time=False):

    start_rgb2gray = None
    rgb2gray_time = 0.
    start_forward_energy = None
    forward_energy_time = 0. 
    start_min_cost = None
    min_cost_time = 0.
    start_min_seam = None
    min_seam_time = 0.
    start_remove_seam = None
    remove_seam_time = 0.

    for i in range(num_remove):
        if test_time:
            start_rgb2gray = time.perf_counter()
        ### convert image to grayscale
        gray_img = rgb2gray(img)
        if test_time:
            rgb2gray_time += time.perf_counter() - start_rgb2gray

        if test_time:
            start_forward_energy = time.perf_counter()
        # calculate energy table
        energy = forward_energy(gray_img)
        if test_time:
            forward_energy_time += time.perf_counter() - start_forward_energy

        if test_time:
            start_min_cost = time.perf_counter()
        # get minimum cost table
        min_costs, backtrack = get_minimum_cost_table(energy)
        if test_time:
            min_cost_time += time.perf_counter() - start_min_cost

        if test_time:
            start_min_seam = time.perf_counter()
        # get minimum seam
        seam_idxs, bool_mask = get_minimum_seam(min_costs, backtrack)
        if test_time:
            min_seam_time += time.perf_counter() - start_min_seam

        if test_time:
            start_remove_seam = time.perf_counter()
        # remove seam
        img = remove_seam(img, seam_idxs)
        if test_time:
            remove_seam_time += time.perf_counter() - start_remove_seam
        

    if test_time: 
        print(f"rgb2gray time: {rgb2gray_time:.3f} seconds")
        print(f"forward energy time:  {forward_energy_time:.3f} seconds")
        print(f"get minimum cost table time: {min_cost_time:.3f} seconds")
        print(f"get minimum seam time:  {min_seam_time:.3f} seconds")
        print(f"remove seam time: {remove_seam_time:.3f} seconds")
    return img


@njit
def insert_seam(img, seam_idx):
    height, width, chanel = img.shape
    output = np.zeros((height, width+1, chanel))
    # The inserted pixel values are derived from an
    # average of left and right neighbors.
    for row in range(height):
        col = seam_idx[row]
        for ch in range(chanel):
            if col == 0:
                p = (img[row, col, ch] + img[row, col+1, ch]) / 2
                output[row, col, ch] = img[row, col, ch]
                output[row, col+1, ch] = p
                output[row, col+1:, ch] = img[row, col:, ch]
            else:
                p = (img[row, col-1, ch] + img[row, col, ch]) / 2
                output[row, :col, ch] = img[row, :col, ch]
                output[row, col, ch] = p
                output[row, col+1:, ch] = img[row, col:, ch]

    return output


def insert_seams(img, num_insert, test_time):
    temp_img = img.copy()  # create replicating image from the input image
    seams_record = []

    start_rgb2gray = None
    rgb2gray_time = 0.
    start_forward_energy = None
    forward_energy_time = 0. 
    start_min_cost = None
    min_cost_time = 0.
    start_min_seam = None
    min_seam_time = 0.
    start_remove_seam = None
    remove_seam_time = 0.
    start_insert_seam = None
    insert_seam_time = 0.

    for _ in range(num_insert):
        if test_time:
            start_rgb2gray = time.perf_counter()
        ### convert image to grayscale
        gray_img = rgb2gray(temp_img)
        if test_time:
            rgb2gray_time += time.perf_counter() - start_rgb2gray

        if test_time:
            start_forward_energy = time.perf_counter()
        # calculate energy table
        energy = forward_energy(gray_img)
        if test_time:
            forward_energy_time += time.perf_counter() - start_forward_energy

        if test_time:
            start_min_cost = time.perf_counter()
        # get minimum cost table
        min_costs, backtrack = get_minimum_cost_table(energy)
        if test_time:
            min_cost_time += time.perf_counter() - start_min_cost

        if test_time:
            start_min_seam = time.perf_counter()
        # get minimum seam
        seam_idxs, bool_mask = get_minimum_seam(min_costs, backtrack)
        if test_time:
            min_seam_time += time.perf_counter() - start_min_seam

        # append seam to insert later
        seams_record.append(seam_idxs)

        if test_time:
            start_remove_seam = time.perf_counter()
        # remove seam
        temp_img = remove_seam(temp_img, seam_idxs)
        if test_time:
            remove_seam_time += time.perf_counter() - start_remove_seam

    seams_record.reverse()
    f = open("cpu.txt", "a")

    for _ in range(num_insert):
        seam = seams_record.pop()


        if test_time:
            start_insert_seam = time.perf_counter()
        # insert seam
        img = insert_seam(img, seam)
        if test_time:
            insert_seam_time += time.perf_counter() - start_insert_seam


        # update remaining seam indices
        for remain_seam in seams_record:
            remain_seam[np.where(remain_seam >= seam)] += 2
        f.write(str(hashlib.sha256(img).hexdigest()) + '\n')


    f.close()
    if test_time:
        print(f"rgb2gray time: {rgb2gray_time:.3f} seconds")
        print(f"forward energy time:  {forward_energy_time:.3f} seconds")
        print(f"get minimum cost table time: {min_cost_time:.3f} seconds")
        print(f"get minimum seam time:  {min_seam_time:.3f} seconds")
        print(f"remove seam time: {remove_seam_time:.3f} seconds")
        print(f"insert seam time: {insert_seam_time:.3f} seconds")


    return img


def seam_carving(img, dx, dy, test_time):
    img = img.astype(np.float64)
    h, w = img.shape[:2]

    assert (h + dy > 0 and w + dx > 0 and dy <= h and dx <= w)

    output = img

    if dx < 0:
        output = remove_seams(output, -dx, test_time)

    elif dx > 0:
        output = insert_seams(output, dx, test_time)

    if dy < 0:
        output = rotate_image(output, True)
        output = remove_seams(output, -dy, test_time)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output = insert_seams(output, dy, test_time)
        output = rotate_image(output, False)

    return output.astype(np.float64)


if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument(
        "-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
    arg_parse.add_argument(
        "-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)

    arg_parse.add_argument("-in", help="Path to image", required=True)
    arg_parse.add_argument("-out", help="Output file name", required=True)
    
    arg_parse.add_argument("-test_time", default=False, action='store_true', help="Test time", required=False)

    args = vars(arg_parse.parse_args())

    print("SEAM CARVING CPU")

    IN_IMG, OUT_IMG = args["in"], args["out"]

    img = cv2.imread(IN_IMG)
    assert img is not None
    print("Input image shape: " + str(img.shape))

    # TODO: resize input images base on dx and dy seam number
    dx, dy = args["dx"], args["dy"]
    assert dx is not None and dy is not None
    start = time.perf_counter()
    output = seam_carving(img, dx, dy, args["test_time"])
    cv2.imwrite(OUT_IMG, output)
    print("Output image shape: " + str(output.shape))
    print(f"Total time: {time.perf_counter() - start:.3f} seconds")
