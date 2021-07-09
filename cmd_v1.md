
# Insert rows

python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert50seams_by_row_baseline.jpg" -dx 0 -dy 50 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert100seams_by_row_baseline.jpg" -dx 0 -dy 100 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert200seams_by_row_baseline.jpg" -dx 0 -dy 200 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert500seams_by_row_baseline.jpg" -dx 0 -dy 500 -checksum
# Insert columns

python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert50seams_by_column_baseline.jpg" -dx 50 -dy 0 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert100seams_by_column_baseline.jpg" -dx 100 -dy 0 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert200seams_by_column_baseline.jpg" -dx 200 -dy 0 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/insert500seams_by_column_baseline.jpg" -dx 500 -dy 0 -checksum

# Remove rows

python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove50seams_by_row_baseline.jpg" -dx 0 -dy -50 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove100seams_by_row_baseline.jpg" -dx 0 -dy -100 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove200seams_by_row_baseline.jpg" -dx 0 -dy -200 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove500seams_by_row_baseline.jpg" -dx 0 -dy -500 -checksum

# Remove columns

python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove50seams_by_column_baseline.jpg" -dx -50 -dy 0 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove100seams_by_column_baseline.jpg" -dx -100 -dy 0 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove200seams_by_column_baseline.jpg" -dx -200 -dy 0 -checksum
python seam_carving_gpu_v1.py -in "images/input.jpg" -out "output_gpu_v1/remove500seams_by_column_baseline.jpg" -dx -500 -dy 0 -checksum