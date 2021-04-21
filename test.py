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

    # def test_energy(self):
    #     img = cv2.imread('images/input.jpg')
    #     output = sc.calc_energy(img)
    #     print(output)

    # def test_get_minimum_seam(self):
    #     img = cv2.imread('images/input.jpg')
    #     img = img.astype(np.float64)
    #     seam_idx, bool_mask = sc.get_minimum_seam(img)
    #     print(seam_idx)

    # def test_remove_seams(self):
    #     img = cv2.imread('images/input.jpg')
    #     print('original image shape: ')
    #     print(img.shape)
    #     num_remove = int(input("input num of seams to remove: "))
    #     output = sc.remove_seams(img, num_remove)
    #     print('new image shape: ')
    #     print(output.shape)
    #     cv2.imwrite('images/output_remove' +str(num_remove) + 'seams.jpg', output)
    #     image = Image.open('images/output_remove' +str(num_remove) + 'seams.jpg')
    #     image.show()
    #     image.close()

    def test_insert_seams(self):
        img = cv2.imread('images/input.jpg')
        print('original image shape: ')
        print(img.shape)
        num_insert = int(input("input num of seams to insert: "))
        output = sc.insert_seams(img, num_insert)
        print('new image shape: ')
        print(output.shape)
        cv2.imwrite('images/output_insert' +str(num_insert) + 'seams.jpg', output)
        image = Image.open('images/output_insert' +str(num_insert) + 'seams.jpg')
        image.show()
        image.close()

    # def test_insert_300seams(self):
    #     img = cv2.imread('images/input.jpg')
    #     img = img.astype(np.float64)
    #     output = sc.insert_seams(img, 300)
    #     cv2.imwrite('images/output_insert300seams.jpg', output)
    #     image = Image.open('images/output_insert300seams.jpg')
    #     image.show()
    #     image.close()


if __name__ == '__main__':
    unittest.main()
