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
	# print(seam_idx)
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


def bench_mark_cpu():
	print("START BENCHMARK CPU")
	start = time.perf_counter()
	test_calc_energy()
	test_get_minimum_seam()
	print("\n")
	num_seams_lst = [50, 100, 200, 300, 500, 900]
	for num_seams in num_seams_lst:
		print("Number of seams: " + str(num_seams))
		test_remove_by_column(num_seams)
		test_remove_by_row(num_seams)
		test_insert_by_column(num_seams)
		test_insert_by_row(num_seams)
		print()
	print(f"Total benchmark time in {time.perf_counter() - start} seconds")

def test_time_cpu():
	img = cv2.imread(INPUT)
	
	print('input image shape: ' + str(img.shape))

	print("rgb2gray time")
	start_rgb2grayscale = time.perf_counter()
	gray_img = sc.rgb2gray(img)
	print(f"Completed convert rgb2gray in {time.perf_counter() - start_rgb2grayscale} seconds")

	print("forward_energy time")
	start_forward_energy = time.perf_counter()
	energy_map = sc.forward_energy(img)
	print(f"Completed forward_energy in {time.perf_counter() - start_forward_energy} seconds")

	print("get_minimum_seam time")
	start_get_seam = time.perf_counter()
	seam_idx, bool_mask = sc.get_minimum_seam(img)
	print(f"Completed get_minimum_seam in {time.perf_counter() - start_get_seam} seconds")

	img2 = img.copy()
	print("remove_seam time")
	start_remove_seam = time.perf_counter()
	output = sc.remove_seams(img, 1)
	print(f"Completed remove_seam in {time.perf_counter() - start_remove_seam} seconds")

	print("insert_seam time")
	start_insert_seam = time.perf_counter()
	output2 = sc.insert_seams(img2, 1)
	print(f"Completed insert_seam in {time.perf_counter() - start_insert_seam} seconds")

def test_forward_energy_parallel(show_img=False):
	print("TEST CALCULATE FORWARD ENERGY PARALLEL")

	start = time.perf_counter()
	img = cv2.imread(INPUT)

	output = sc.forward_energy_parallel(img)

	cv2.imwrite('images/ouput_forward_energy_parallel.jpg', output)
	if show_img:
		image = Image.open('images/ouput_forward_energy_parallel.jpg')
		image.show()
	print(f"Completed Execution in {time.perf_counter() - start} seconds")


if __name__ == '__main__':
	# test_calc_energy()
	#test_forward_energy()
	test_forward_energy_parallel()
	# test_get_minimum_seam()
	#test_remove_by_column(num_seams=900)
	# test_remove_by_row(num_seams=200)
	# test_insert_by_column(num_seams=500)
	# test_insert_by_row(num_seams=500)

	# bench_mark_cpu()
	
	# test_time_cpu()