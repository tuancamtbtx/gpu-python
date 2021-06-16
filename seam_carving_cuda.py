import PIL.Image
from io import BytesIO
import IPython.display

import cv2
import numpy as np
from numba import jit, njit, cuda

import argparse

# ignore numba warning
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning, NumbaWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaWarning)



@cuda.jit
def rgb2gray_kernel(rgb_img, gray_img):
    i, j = cuda.grid(2)

    if i < rgb_img.shape[0] and j < rgb_img.shape[1]:
        gray_img[i,j] = 0.2989*rgb_img[i,j,0] + 0.5870*rgb_img[i,j,1] + 0.1140*rgb_img[i,j,2]
    return

def rotate_image(img, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(img, k)


@njit
def forward_energy(img):
    height, width = img.shape[:2]
    # img = rgb2gray(img)

    energy = np.zeros((height, width))  # energy table
    m = np.zeros((height, width))

    U = np.empty(img.shape, dtype=np.float64)
    for row in range(height):
        if row == height - 1:
            U[0] = img[row]
            break
        U[row + 1] = img[row]

    L = np.empty(img.shape, dtype=np.float64)
    for col in range(width):
        if col == width - 1:
            L[:, 0] = img[:, col]
            break
        L[:, col + 1] = img[:, col]

    R = np.empty(img.shape, dtype=np.float64)
    for col in range(width):
        if col == width-1:
            R[:, col] = img[:, 0]
            break
        R[:,col - 1] = img[:, col]

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, height):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)

        mULR = np.empty((3, width), dtype=np.float64)
        mULR[0] = mU
        mULR[1] = mL
        mULR[2] = mR

        cULR = np.empty((3, width), dtype=np.float64)
        cULR[0] = cU[i]
        cULR[1] = cL[i]
        cULR[2] = cR[i]

        mULR += cULR

        # implement np argmin for numba
        argmins = np.empty(mULR.shape[1], dtype=np.int8)
        for j in range(mULR.shape[1]):
            min_val =  mULR[0][j]
            min_idx = 0
            for k in range(1, 3):
                if min_val > mULR[k][j]:
                    min_val = mULR[k][j]
                    min_idx = k

            argmins[j] = min_idx

        for idx in range(len(argmins)):
            m[i][idx] = mULR[argmins[idx]][idx]

        for idx in range(len(argmins)):
            energy[i][idx] = cULR[argmins[idx]][idx]

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

@cuda.jit
def remove_seam_kernel(img, seam_idxs, out_img):
    i, j = cuda.grid(2)

    if i < img.shape[0] and j < img.shape[1]:
        for ch in range(img.shape[2]):
            if seam_idxs[i] <= j:
                out_img[i][j][ch] = img[i][j + 1][ch]
            else:
                out_img[i][j][ch] = img[i][j][ch]
    return

def remove_seams_kernel(img, num_remove, rot=False):
    in_img = img.copy()

    ### convert rgb to grayscale
    #Run kernel
    griddim = 200, 230
    blockdim = 32, 32

    for _ in range(num_remove):
        # Send to GPU
        d_img = cuda.to_device(in_img)
        d_gray_out = cuda.device_array(in_img.shape[0:2])

        rgb2gray_kernel[griddim, blockdim](d_img, d_gray_out)

        gray_img = np.asarray(d_gray_out)

        ### get minimum seam
        seam_idxs , bool_mask = get_minimum_seam(gray_img)

        ### remove seam
        # Send to GPU
        d_bool_mask = cuda.to_device(bool_mask)
        d_seam_idxs = cuda.to_device(seam_idxs)
        # d_bool_mask_index = cuda.device_array((in_img.shape[0], 1))
        d_out = cuda.device_array((in_img.shape[0], in_img.shape[1] - 1, in_img.shape[2]))
        # d_out = cuda.device_array((in_img.shape[0], in_img.shape[1], in_img.shape[2]))

        # find_seams_index_by_rows[griddim, blockdim](d_bool_mask, d_bool_mask_index)

        remove_seam_kernel[griddim, blockdim](d_img, d_seam_idxs, d_out)

        in_img = np.asarray(d_out).astype(np.uint8)

    print(in_img.shape)
    return in_img

@cuda.jit
def insert_seam_kernel(img, seam_idxs, out_img):
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


def insert_seams_kernel(img, num_insert):
    temp_img = img.copy()  # create replicating image from the input image
    seams_record = []
    for _ in range(num_insert):
        seam, bool_mask = get_minimum_seam(temp_img)
        seams_record.append(seam)
        temp_img = remove_seams_kernel(temp_img, bool_mask)

    seams_record.reverse()

    for _ in range(num_insert):
        seam = seams_record.pop()
        img = insert_seam_kernel(img, seam)

        # update remaining seam indices
        for remain_seam in seams_record:
            remain_seam[np.where(remain_seam >= seam)] += 2

    return img

def seam_carving_kernel(img, dx, dy):
    img = img.astype(np.float64)
    h, w = img.shape[:2]

    assert (h + dy > 0 and w + dx > 0 and dy <= h and dx <= w)

    output = img

    if dx < 0:
        output = remove_seams_kernel(output, -dx)

    elif dx > 0:
        output = insert_seams_kernel(output, dx)

    if dy < 0:
        output = rotate_image(output, True)
        output = remove_seams_kernel(output, -dy, rot=True)
        output = rotate_image(output, False)

    elif dy > 0:
        output = rotate_image(output, True)
        output = insert_seams_kernel(output, dy, rot=True)
        output = rotate_image(output, False)

    return output