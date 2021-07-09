import sys
import os
import hashlib
import cv2

from os import listdir, write
from os.path import isfile, join

def is_2_images_equal(img1, img2):
    hash_img1 =str(hashlib.sha256(img1).hexdigest())
    hash_img2 =str(hashlib.sha256(img2).hexdigest())

    return hash_img1 == hash_img2, hash_img1, hash_img2

def compare_cpu_and_gpu(path1='output_cpu', path2='output_gpu', write_to_file=True):
    """compare_cpu_and_gpu

    Compare all images in 2 folders of cpu vs gpu. 
    All names must be the same between images

    Args:
        path1 (str): path to folder 1
        path2 (str): path to folder 2
    """
    assert path1 != path2
    path1_filenames = [f for f in listdir(path1) if isfile(join(path1, f))]
    path2_filenames = [f for f in listdir(path2) if isfile(join(path2, f))]

    f = open("compare_images_result.txt", "w")
    total_pass = 0
    total_fail = 0
    for filename1 in path1_filenames:
        for filename2 in path2_filenames:
            if (filename1 == filename2):
                path_file1 = os.path.join(path1, filename1)
                path_file2 = os.path.join(path2, filename2)

                img1 = cv2.imread(path_file1)
                img2 = cv2.imread(path_file2)

                is_equal, hash_img1, hash_img2 = is_2_images_equal(img1, img2)
                status = 'PASS' if is_equal else 'FAIL'

                if is_equal:
                    total_pass = total_pass + 1
                else:
                    total_fail = total_fail + 1

                print('Comparing {0} & {1}'.format(path_file1, path_file2))
                print('Status: {0}'.format(status))
                print('Image 1(sha256): {0}'.format(hash_img1))
                print('Image 2(sha256): {0}'.format(hash_img2))

                if write_to_file:
                    f.write('Comparing {0} & {1}\n'.format(path_file1, path_file2))
                    f.write('Status: {0}\n'.format(status))
                    f.write('Image 1(sha256): {0}\n'.format(hash_img1))
                    f.write('Image 2(sha256): {0}\n\n'.format(hash_img2))

    print('Total PASS: {0}'.format(total_pass))
    print('Total FAIL: {0}'.format(total_fail))

    if write_to_file:
        f.write('\nTotal PASS: {0}\n'.format(total_pass))
        f.write('\nTotal FAIL: {0}\n'.format(total_fail))

    f.close()

def main():
    path1, path2 = sys.argv[1:1+2]
    compare_cpu_and_gpu(path1=path1, path2=path2)

if __name__ == "__main__":
    main()