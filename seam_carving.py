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



# @cuda.jit
# def rgb2gray(rgb_img, gray_img):
#     i, j =  cuda.grid(2)

#     if i < rgb_img.shape[0] and j < rgb_img.shape[1]:
#         gray_img[i,j] = 0.2989*rgb_img[i,j,0] + 0.5870*rgb_img[i,j,1] + 0.1140*rgb_img[i,j,2]
#     return

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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(img.shape)

    if args["mode"] == "gpu":
        # TODO: resize input images base on dx and dy seam number
        # dx, dy = args["dx"], args["dy"]
        # assert dx is not None and dy is not None

        # #Run kernel
        # griddim = 200, 230
        # blockdim = 16, 16

        # # Send to GPU
        # d_img = cuda.to_device(img)
        # d_out = cuda.device_array(img.shape[0:2])

        # rgb2gray[griddim, blockdim](d_img, d_out)

        # out_img =  PIL.Image.fromarray(np.asarray(d_out).astype(np.uint8), 'L')
        # out_img.save(OUT_IMG)
        # test command python seam_carving.py -mode=gpu -in=images/input.jpg -out=images/input_grayscal.jpg
        # or python test.py
        pass

    elif args["mode"] == "cpu":
        # TODO: resize input images base on dx and dy seam number
        dx, dy = args["dx"], args["dy"]
        assert dx is not None and dy is not None
        output = seam_carving(img, dx, dy)
        cv2.imwrite(OUT_IMG, output)