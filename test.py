import unittest
import seam_carving as sc
import cv2
import numpy as np
from PIL import Image


class Test(unittest.TestCase):

    # def test_sobel(self):
    #     img = cv2.imread('images/input.jpg')
    #     output = sc.calc_energy(img)
    #     cv2.imwrite('images/output_sobel.jpg', output)
    #     image = Image.open('images/output_sobel.jpg')
    #     image.show()
    #     image.close()
    
    # def test_get_minimum_seam(self):
    #     img = cv2.imread('images/input.jpg')
    #     seam_idx, bool_mask = sc.get_minimum_seam(img)
    #     print(seam_idx)

    def test_remove_seams(self):
        img = cv2.imread('images/input.jpg')
        img = img.astype(np.float64)
        output = sc.remove_seams(img, 100)
        cv2.imwrite('images/output_remove100seams.jpg', output)
        image = Image.open('images/output_remove100seams.jpg')
        image.show()
        image.close()

        
if __name__ == '__main__':
    unittest.main()
