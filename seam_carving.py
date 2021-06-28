from PIL import Image
import cv2
import numpy as np
from numba import jit, njit, cuda
import math

import argparse

# ignore numba warning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

MAX_THREADS = 1024


@njit
def rgb2gray(img):
    img = img.astype(np.float64)
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
def old_forward_energy(img):
    height, width, _ = img.shape

    # convert rgb image to grayscale
    img = rgb2gray(img)

    # energy map
    energy = np.zeros((height, width))
    # m table for dynamic programming
    m = np.zeros((height, width))

    # implement U = np.roll(img, 1, axis=0) for numba
    U = np.empty(img.shape, dtype=np.float64)
    for row in range(height):
        if row == height - 1:
            U[0] = img[row]
            break
        U[row + 1] = img[row]

    # implement L = np.roll(img, 1, axis=1) for numba
    L = np.empty(img.shape, dtype=np.float64)
    for col in range(width):
        if col == width - 1:
            L[:, 0] = img[:, col]
            break
        L[:, col + 1] = img[:, col]

    # implement R = np.roll(img, -1, axis=1) for numba
    R = np.empty(img.shape, dtype=np.float64)
    for col in range(width):
        if col == width-1:
            R[:, col] = img[:, 0]
            break
        R[:, col - 1] = img[:, col]

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, height):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        # implement mULR = np.array([mU, mL, mR]) for numba
        mULR = np.empty((3, width), dtype=np.float64)
        mULR[0] = mU
        mULR[1] = mL
        mULR[2] = mR

        # implement cULR = np.array([cU[i], cL[i], cR[i]]) for numba
        cULR = np.empty((3, width), dtype=np.float64)
        cULR[0] = cU[i]
        cULR[1] = cL[i]
        cULR[2] = cR[i]

        mULR += cULR

        # implement argmins = np.argmin(mULR, axis=0) for numba
        argmins = np.empty(mULR.shape[1], dtype=np.int8)
        for j in range(mULR.shape[1]):
            min_val = mULR[0][j]
            min_idx = 0
            for k in range(1, 3):
                if min_val > mULR[k][j]:
                    min_val = mULR[k][j]
                    min_idx = k

            argmins[j] = min_idx

        # implement m[i] = np.choose(argmins, mULR) for numba
        for idx in range(len(argmins)):
            m[i][idx] = mULR[argmins[idx]][idx]

        # choose the lowest-energy option.
        # implement energy[i] = np.choose(argmins, cULR) for numba
        for idx in range(len(argmins)):
            energy[i][idx] = cULR[argmins[idx]][idx]

    return energy


@njit
def forward_energy(img):
    height, width = img.shape[:2]

    gray_img = rgb2gray(img)
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


@cuda.jit
def forward_energy_kernel(in_pixels, r, m, energy):
    c = cuda.grid(1)
    height, width = in_pixels.shape[:2]
    if c < width:
        up = (r-1) % height
        left = (c-1) % width
        right = (c+1) % width

        mU = m[up, c]
        mL = m[up, left]
        mR = m[up, right]

        cU = abs(in_pixels[r, right] - in_pixels[r, left])
        cL = abs(in_pixels[up, c] - in_pixels[r, left]) + cU
        cR = abs(in_pixels[up, c] - in_pixels[r, right]) + cU

        mU += cU
        mL += cL
        mR += cR

        _min = mU
        _cMin = cU
        if mL < _min:
            _min = mL
            _cMin = cL
        elif mR < _min:
            _min = mR
            _cMin = cR
        m[r, c] = _min
        energy[r, c] = _cMin


def forward_energy_parallel(img):
    height, width = img.shape[:2]

    gray_img = rgb2gray(img)

    energy = np.zeros((height, width))  # energy table

    d_gray_img = cuda.to_device(gray_img)
    d_energy = cuda.device_array((height, width), dtype=np.float64)
    d_m = cuda.device_array((height, width), dtype=np.float64)

    block_size = 32
    grid_size = math.ceil(width / block_size)

    for r in range(height):
        # calculate forward energy kernel row by row
        forward_energy_kernel[block_size, grid_size](
            d_gray_img, r, d_m, d_energy)

    energy = d_energy.copy_to_host()
    return energy


@jit
def get_minimum_seam(img):
    h, w = img.shape[:2]

    # matrix to store minimum energy value seen upon pixel
    M = forward_energy(img)

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
def get_minimum_seam_parallel(img):
    h, w = img.shape[:2]

    # convert rgb2gray kernel
    gray_img = rgb2gray(img)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    d_gray_img = cuda.to_device(gray_img)
    d_energy = cuda.to_device(energy)
    d_m = cuda.to_device(m)

    block_size = (32, 32)
    grid_size = (math.ceil(h / block_size[0]),
                 math.ceil(w / block_size[1]))
    M = forward_energy_kernel[grid_size, block_size](
        d_gray_img, d_energy, d_m, h, w)

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


@njit
def remove_seam(img, bool_mask):
    height, width, chanel = img.shape

    output = np.empty((height, width-1, chanel), dtype=np.float64)
    for row in range(height):
        output_col = 0
        for col in range(width):
            if bool_mask[row][col]:
                for ch in range(chanel):
                    output[row][output_col][ch] = img[row][col][ch]
                output_col += 1
            else:
                continue

    return output


@jit
def remove_seams(img, num_remove, rot=False):
    for _ in range(num_remove):
        _, bool_mask = get_minimum_seam(img)
        img = remove_seam(img, bool_mask)

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


def insert_seams(img, num_insert):
    temp_img = img.copy()  # create replicating image from the input image
    seams_record = []
    for _ in range(num_insert):
        seam, bool_mask = get_minimum_seam(temp_img)
        seams_record.append(seam)
        temp_img = remove_seam(temp_img, bool_mask)

    seams_record.reverse()

    for _ in range(num_insert):
        seam = seams_record.pop()
        img = insert_seam(img, seam)

        # update remaining seam indices
        for remain_seam in seams_record:
            remain_seam[np.where(remain_seam >= seam)] += 2

    return img


def seam_carving(img, dx, dy):
    img = img.astype(np.float64)
    h, w = img.shape[:2]

    assert (h + dy > 0 and w + dx > 0 and dy <= h and dx <= w)

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


if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument(
        "-mode", help="Type of running seam: cpu or gpu", type=str, default=0)

    arg_parse.add_argument(
        "-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
    arg_parse.add_argument(
        "-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)

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
