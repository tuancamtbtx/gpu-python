import seam_carving as sc
import cv2
import numpy as np
from PIL import Image
import time

INPUT = 'images/input.jpg'


def test_calc_energy(show_img=False):
	print("TEST CALCULATE ENERGY")

	start = time.perf_counter()
	img = cv2.imread(INPUT)

	output = sc.calc_energy(img)

	cv2.imwrite('images/ouput_energy.jpg', output)
	if show_img:
		image = Image.open('images/ouput_energy.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")

def test_forward_energy(show_img=False):
	print("TEST CALCULATE FORWARD ENERGY")

	start = time.perf_counter()
	img = cv2.imread(INPUT)

	output = sc.forward_energy(img)

	cv2.imwrite('images/ouput_forward_energy.jpg', output)
	if show_img:
		image = Image.open('images/ouput_forward_energy.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")


def test_get_minimum_seam():
	print("TEST GET MINIMUM SEAM")
	start = time.perf_counter()
	img = cv2.imread(INPUT)
	img = img.astype(np.float64)
	seam_idx, bool_mask = sc.get_minimum_seam(img)
	print(seam_idx)
	print(f"Completed Execution in {time.perf_counter() - start} seconds")

def test_remove_by_column(num_seams=10, show_img=False):
	print("TEST REMOVE BY COLUMN")
	print("number of seams to remove: " + str(num_seams))

	start = time.perf_counter()
	img = cv2.imread(INPUT)
	print('input image shape: ' + str(img.shape))

	output = sc.remove_seams(img, num_seams)
	print('output image shape: ' + str(output.shape))

	cv2.imwrite('images/output_remove' +str(num_seams) + 'seams_by_column.jpg', output)
	if show_img:
		image = Image.open('images/output_remove' +str(num_seams) + 'seams_by_column.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")



def test_remove_by_row(num_seams=10, show_img=False):
	print("TEST REMOVE BY ROW")
	print("number of seams to remove: " + str(num_seams))

	start = time.perf_counter()
	img = cv2.imread(INPUT)
	print('input image shape: ' + str(img.shape))

	output = img
	output = sc.rotate_image(output, True)
	output = sc.remove_seams(output, num_seams)
	output = sc.rotate_image(output, False) 
	print('output image shape: ' + str(output.shape))

	cv2.imwrite('images/output_remove' +str(num_seams) + 'seams_by_row.jpg', output)
	if show_img:
		image = Image.open('images/output_remove' +str(num_seams) + 'seams_by_row.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")

def test_insert_by_column(num_seams=10, show_img=False):
	print("TEST INSERT SEAM BY COLUMN")
	print("number of seams to insert: " + str(num_seams))

	start = time.perf_counter()
	img = cv2.imread(INPUT)
	print('input image shape: ' + str(img.shape))

	output = sc.insert_seams(img, num_seams)
	print('output image shape: ' + str(output.shape))

	cv2.imwrite('images/output_insert' +str(num_seams) + 'seams_by_column.jpg', output)
	if show_img:
		image = Image.open('images/output_insert' +str(num_seams) + 'seams_by_column.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")

def test_insert_by_row(num_seams=10, show_img=False):
	print("TEST INSERT SEAM BY ROW")
	print("number of seams to insert: " + str(num_seams))

	start = time.perf_counter()

	img = cv2.imread(INPUT)
	print('input image shape: ' + str(img.shape))

	output = img
	output = sc.rotate_image(output, True)
	output = sc.insert_seams(output, num_seams)
	output = sc.rotate_image(output, False)
	print('new image shape: ' + str(output.shape))

	cv2.imwrite('images/output_insert' +str(num_seams) + 'seams_by_rows.jpg', output)
	if show_img:
		image = Image.open('images/output_insert' +str(num_seams) + 'seams_by_rows.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")


if __name__ == '__main__':
	# test_calc_energy()
	# test_forward_energy()
	# test_get_minimum_seam()
	# test_remove_by_column(num_seams=100)
	# test_remove_by_row(num_seams=100)
	test_insert_by_column(num_seams=500)
	# test_insert_by_row(num_seams=500)
	
