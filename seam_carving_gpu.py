import cv2
import numpy as np
from numba import jit, njit, cuda, float64, int16
import math
import time
import argparse

# ignore numba warning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)

MAX_WIDTH = 1500

@njit
def rgb2gray(img):
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
    # return np.rot90(img, k)



@cuda.jit
def rgb2gray_kernel(rgb_img, gray_img):
    i, j = cuda.grid(2)
    if i < rgb_img.shape[0] and j < rgb_img.shape[1]:
        curr_pixel = rgb_img[i, j]
        gray_img[i, j] = 0.2989*curr_pixel[0] + \
            0.5870*curr_pixel[1] + 0.1140*curr_pixel[2]


@cuda.jit
def forward_energy_kernel(gray_img, energy, m, r):
    c = cuda.grid(1)
    height, width = gray_img.shape[:2]
    if c < width:
        up = (r-1) % height
        left = (c-1) % width
        right = (c+1) % width

        mU = m[up, c]
        mL = m[up, left]
        mR = m[up, right]

        cU = abs(gray_img[r, right] - gray_img[r, left])
        cL = abs(gray_img[up, c] - gray_img[r, left]) + cU
        cR = abs(gray_img[up, c] - gray_img[r, right]) + cU

        mU += cU
        mL += cL
        mR += cR

        _mMin = mU
        _cMin = cU
        if mL < _mMin:
            _mMin = mL
            _cMin = cL
        if mR < _mMin:
            _mMin = mR
            _cMin = cR

        m[r, c] = _mMin
        energy[r, c] = _cMin


@cuda.jit
def get_minimum_cost_table_kernel(energy, backtrack, w, r):
    c = cuda.grid(1)
    if (c < w):
        left = max(0, c - 1)
        right = min(w - 1, c + 1)

        minimum = energy[r-1, left]
        backtrack_col = left
        if energy[r-1, c] < minimum:
            minimum = energy[r-1, c]
            backtrack_col = c
        if energy[r-1, right] < minimum:
            minimum = energy[r-1, right]
            backtrack_col = right

        energy[r, c] += minimum
        backtrack[r, c] = backtrack_col


@njit
def get_minimum_seam(min_costs, backtrack):
    h, w = min_costs.shape[:2]
    seam_idx = []
    bool_mask = np.ones((h, w), dtype=np.bool8)
    # find the minum cost in bottom row
    j = np.argmin(min_costs[-1])
    for i in range(h-1, -1, -1):
        bool_mask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx = [seam_idx[-i - 1] for i in range(len(seam_idx))]
    return np.array(seam_idx), bool_mask


@cuda.jit
def remove_seam_kernel(img, seam_idxs, height, width, out_img):
    i, j = cuda.grid(2)

    if i < height and j < width:
        if seam_idxs[i] <= j:
            for ch in range(img.shape[2]):
                out_img[i][j][ch] = img[i][j + 1][ch]
        else:
            for ch in range(img.shape[2]):
                out_img[i][j][ch] = img[i][j][ch]
    return


def remove_seams_parallel(img, num_remove, checksum=False):

    for _ in range(num_remove):
        height, width = img.shape[:2]
        block_size = 32, 32
        grid_size = (math.ceil(height / block_size[0]),
                    math.ceil(width / block_size[1]))
        
        d_img = cuda.to_device(img)

        gray_img = None
        d_gray_img = None
        if checksum:
            gray_img = rgb2gray(img)
            d_gray_img = cuda.to_device(gray_img)
        else:
            ### gray2rgb kernel
            d_gray_img = cuda.device_array(img.shape[0:2], dtype=np.float64)
            rgb2gray_kernel[grid_size, block_size](d_img, d_gray_img)


        ### forward energy kernel
        d_energy = cuda.device_array((height, width), dtype=np.float64)
        d_m = cuda.device_array((height, width), dtype=np.float64)

        block_size_energy = 32
        grid_size_energy = math.ceil(width / block_size_energy)

        for r in range(1, height):
            forward_energy_kernel[grid_size_energy, block_size_energy](
                d_gray_img, d_energy, d_m, r)


        ### get minimum cost table kernel
        d_backtrack = cuda.device_array((height, width), dtype=np.uint16)

        block_size_min_cost = 32
        grid_size_min_cost = math.ceil(width / block_size_min_cost)
        for r in range(1, height):
            get_minimum_cost_table_kernel[grid_size_min_cost, block_size_min_cost](
                d_energy, d_backtrack, width, r)
        min_costs = d_energy.copy_to_host()
        backtrack = d_backtrack.copy_to_host()

        ### get minimum seam
        seam_idxs, bool_mask = get_minimum_seam(min_costs, backtrack)

        ### remove seam kernel
        d_seam_idxs = cuda.to_device(seam_idxs)
        d_out = cuda.device_array((height, width - 1, img.shape[2]))

        remove_seam_kernel[grid_size, block_size](
            d_img, d_seam_idxs, height, width - 1, d_out)

        img = d_out.copy_to_host()

    return img


@cuda.jit
def insert_seam_kernel(img, seam_idxs, height, width, out_img):
    # The inserted pixel values are derived from an
    # average of left and right neighbors.
    row, col = cuda.grid(2)
    if row < height and col < width:
        if seam_idxs[row] == col:
            if col == 0:
                for ch in range(3):
                    p = (img[row, col, ch] + img[row, col + 1, ch]) / 2
                    out_img[row][col][ch] = p
                    out_img[row][col + 1][ch] = img[row][col][ch]
            else:
                for ch in range(3):
                    p = (img[row, col - 1, ch] + img[row, col, ch]) / 2
                    out_img[row][col][ch] = p
                    out_img[row][col + 1][ch] = img[row][col][ch]

        elif col < seam_idxs[row]:
            for ch in range(3):
                out_img[row][col][ch] = img[row][col][ch]
        else:
            for ch in range(3):
                out_img[row][col + 1][ch] = img[row][col][ch]

    return


def insert_seams_parallel(img, num_insert, checksum=False):
    temp_img = img.copy()  # create replicating image from the input image
    seams_record = []

    for _ in range(num_insert):
        height, width = temp_img.shape[:2]
        block_size = 32, 32
        grid_size = (math.ceil(height / block_size[0]),
                    math.ceil(width / block_size[1]))

        d_img = cuda.to_device(temp_img)
        gray_img = None
        d_gray_img = None
        if checksum:
            gray_img = rgb2gray(temp_img)
            d_gray_img = cuda.to_device(gray_img)
        else:
            ## gray2rgb kernel
            d_gray_img = cuda.device_array(temp_img.shape[0:2])
            rgb2gray_kernel[grid_size, block_size](d_img, d_gray_img)


        # forward energy kernel
        d_energy = cuda.device_array((height, width), dtype=np.float64)
        d_m = cuda.device_array((height, width), dtype=np.float64)

        block_size_energy = 32
        grid_size_energy = math.ceil(width / block_size_energy)

        for r in range(1, height):
            forward_energy_kernel[grid_size_energy, block_size_energy](
                d_gray_img, d_energy, d_m, r)

        # get minimum cost table kernel
        d_backtrack = cuda.device_array((height, width), dtype=np.uint16)

        block_size_min_cost = 32
        grid_size_min_cost = math.ceil(width / block_size_min_cost)

        for r in range(1, height):
            get_minimum_cost_table_kernel[grid_size_min_cost, block_size_min_cost](
                d_energy, d_backtrack, width, r)

        min_costs = d_energy.copy_to_host()
        backtrack = d_backtrack.copy_to_host()
        seam_idxs, bool_mask = get_minimum_seam(min_costs, backtrack)

        # remove seam kernel
        d_seam_idxs = cuda.to_device(seam_idxs)
        d_out = cuda.device_array((height, width - 1, 3), dtype=np.float64)

        remove_seam_kernel[grid_size, block_size](
            d_img, d_seam_idxs, height, width - 1, d_out)

        temp_img = d_out.copy_to_host()

		# append seam to insert later
        seams_record.append(d_seam_idxs.copy_to_host())


    seams_record.reverse()
    for _ in range(num_insert):
        height, width = img.shape[:2]
        seam_idxs = seams_record.pop()

        block_size = 32, 32
        grid_size = (math.ceil(height / block_size[0]),
                    math.ceil(width / block_size[1]))

        # insert seam kernel
        d_seam_idxs = cuda.to_device(seam_idxs)
        d_img = cuda.to_device(img)
        d_out = cuda.device_array((height, width + 1, img.shape[2]))
        insert_seam_kernel[grid_size, block_size](
            d_img, d_seam_idxs, height, width, d_out)
        img = d_out.copy_to_host()

        # update remaining seam indices
        for remain_seam in seams_record:
            remain_seam[np.where(remain_seam >= seam_idxs)] += 2
    return img


def seam_carving(img, dx, dy, debug):
    img = img.astype(np.float64)
    h, w = img.shape[:2]

    assert (h + dy > 0 and w + dx > 0 and dy <= h and dx <= w)

    output = img

    if dx < 0:
        output = remove_seams_parallel(output, -dx, debug)

    elif dx > 0:
        output = insert_seams_parallel(output, dx, debug)

    if dy < 0:
        output = rotate_image(output, True)
        output = remove_seams_parallel(output, -dy, debug)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output = insert_seams_parallel(output, dy, debug)
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

    arg_parse.add_argument("-checksum", default=False, action='store_true', help="Calculate rgb2gray by host for SHASUM", required=False)

    args = vars(arg_parse.parse_args())

    print("SEAM CARVING GPU - BASELINE")

    IN_IMG, OUT_IMG = args["in"], args["out"]

    img = cv2.imread(IN_IMG)
    assert img is not None
    print("Input image shape: " + str(img.shape))

    # TODO: resize input images base on dx and dy seam number
    dx, dy = args["dx"], args["dy"]
    assert dx is not None and dy is not None
    start = time.perf_counter()
    output = seam_carving(img, dx, dy, args["checksum"])
    cv2.imwrite(OUT_IMG, output)
    print("Output image shape: " + str(output.shape))
    print(f"Total time: {time.perf_counter() - start:.3f} seconds")
    # import seam_carving_cpu as cpu
    # output2 = cpu.seam_carving(img, dx, dy, False)
    # for i in range(output.shape[0]):
    #     for j in range(output.shape[1]):
    #         if (output[i, j] != output2[i, j]).any():
    #             print(output[i, j], output2[i, j])
    # print(hash(str(output)))
    # print(hash(str(output2)))
    # print(np.mean(np.abs(output - output2)))
