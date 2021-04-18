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


def visualize_image(img, boolmask=None, rotate=False):
    visuallize = img.astype(np.uint8)
    if boolmask is not None:
        visuallize[np.where(boolmask == False)] = np.array([255, 200, 200]) # BGR

    cv2.imshow("visualization", visuallize)
    cv2.waitKey(1)
    return visuallize


def calc_energy(img, filter_dx=FILTER_DX, filter_dy=FILTER_DY):
    """
    Simple gradient magnitude energy map.
    """
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dxs = np.stack([filter_dx] * 3, axis=2)
    filter_dys = np.stack([filter_dy] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_dxs)) + np.absolute(convolve(img, filter_dys))

    # We sum the energies in the red, green, and blue channels
    grad_mag_map = convolved.sum(axis=2)

    return grad_mag_map


if __name__ == '__main__':
    arg_parse = argparse.ArgumentParser()

    arg_parse.add_argument("-mode", help="Type of running seam: cpu or gpu", type=int, default=0)

    arg_parse.add_argument("-dy", help="Number of vertical seams to add/subtract", type=int, default=0)
    arg_parse.add_argument("-dx", help="Number of horizontal seams to add/subtract", type=int, default=0)

    arg_parse.add_argument("-in", help="Path to image", required=True)
    arg_parse.add_argument("-out", help="Output file name", required=True)


    args = vars(arg_parse.parse_args())


    if args["mode"] == "gpu":
        # TODO: later
        pass
    
    elif args["mode"] == "cpu":
        # TODO: resize input images base on dx and dy seam number
        
        pass
