import numpy as np
from numba import jit
import cv2
import argparse




def visualize_image(im, boolmask=None, rotate=False):
    visuallize = im.astype(np.uint8)
    if boolmask is not None:
        visuallize[np.where(boolmask == False)] = np.array([255, 200, 200]) # BGR

    cv2.imshow("visualization", visuallize)
    cv2.waitKey(1)
    return visuallize



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
