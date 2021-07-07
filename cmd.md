
# Insert rows
## GPU
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert50seams_by_row_baseline.jpg" -dx 0 -dy 50
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert100seams_by_row_baseline.jpg" -dx 0 -dy 100
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert200seams_by_row_baseline.jpg" -dx 0 -dy 200
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert500seams_by_row_baseline.jpg" -dx 0 -dy 500

## CPU
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert50seams_by_row_baseline.jpg" -dx 0 -dy 50
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert100seams_by_row_baseline.jpg" -dx 0 -dy 100
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert200seams_by_row_baseline.jpg" -dx 0 -dy 200
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert500seams_by_row_baseline.jpg" -dx 0 -dy 500

# Insert columns
## GPU
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert50seams_by_column_baseline.jpg" -dx 50 -dy 0
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert100seams_by_column_baseline.jpg" -dx 100 -dy 0
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert200seams_by_column_baseline.jpg" -dx 200 -dy 0
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/insert500seams_by_column_baseline.jpg" -dx 500 -dy 0

## CPU
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert50seams_by_column_baseline.jpg" -dx 50 -dy 0
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert100seams_by_column_baseline.jpg" -dx 100 -dy 0
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/insert200seams_by_column_baseline.jpg" -dx 200 -dy 0
python seam_carving_cpu.py -in "images/input.jpg"c-out "output_cpu/insert500seams_by_column_baseline.jpg" -dx 500 -dy 0

# Remove rows
## GPU
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove50seams_by_row_baseline.jpg" -dx 0 -dy -50
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove100seams_by_row_baseline.jpg" -dx 0 -dy -100
python seam_carving_gpu.py -in "images/input.j-pg" -out "output_gpu/remove200seams_by_row_baseline.jpg" -dx 0 -dy -200
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove500seams_by_row_baseline.jpg" -dx 0 -dy -500

## CPU
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove50seams_by_row_baseline.jpg" -dx 0 -dy -50
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove100seams_by_row_baseline.jpg" -dx 0 -dy -100
python seam_carving_cpu.py -in "images/input.j-pg" -out "output_cpu/remove200seams_by_row_baseline.jpg" -dx 0 -dy -200
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove500seams_by_row_baseline.jpg" -dx 0 -dy -500

# Remove columns
## GPU
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove50seams_by_column_baseline.jpg" -dx -50 -dy 0
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove100seams_by_column_baseline.jpg" -dx -100 -dy 0
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove200seams_by_column_baseline.jpg" -dx -200 -dy 0
python seam_carving_gpu.py -in "images/input.jpg" -out "output_gpu/remove500seams_by_column_baseline.jpg" -dx -500 -dy 0

## CPU
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove50seams_by_column_baseline.jpg" -dx -50 -dy 0
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove100seams_by_column_baseline.jpg" -dx -100 -dy 0
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove200seams_by_column_baseline.jpg" -dx -200 -dy 0
python seam_carving_cpu.py -in "images/input.jpg" -out "output_cpu/remove500seams_by_column_baseline.jpg" -dx -500 -dy 0