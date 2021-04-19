import unittest
import seam_carving as sc
import cv2
import numpy as np
from PIL import Image


class Test(unittest.TestCase):

    def test_sobel(self):
        img = cv2.imread('images/input.jpg')
        output = sc.calc_energy(img)
        cv2.imwrite('images/output_sobel.jpg', output)
        image = Image.open('images/output_sobel.jpg')
        image.show()

if __name__ == '__main__':
    unittest.main()
