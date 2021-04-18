import numpy as np
from numba import jit

import argparse





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