# crowded_insert rows
Reference: https://www.verywellmind.com/an-overview-of-enochlophobia-4782189
## GPU
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert20seams_by_row_baseline.jpg" -dx 0 -dy 20 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert50seams_by_row_baseline.jpg" -dx 0 -dy 50 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert100seams_by_row_baseline.jpg" -dx 0 -dy 100 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert200seams_by_row_baseline.jpg" -dx 0 -dy 200 -checksu
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert200seams_by_row_and_column_baseline.jpg" -dx 200 -dy 200 -checksum

## CPU
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert20seams_by_row_baseline.jpg" -dx 0 -dy 20
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert50seams_by_row_baseline.jpg" -dx 0 -dy 50
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert100seams_by_row_baseline.jpg" -dx 0 -dy 100
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert200seams_by_row_baseline.jpg" -dx 0 -dy 200
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert200seams_by_row_and_column_baseline.jpg" -dx 200 -dy 200


# crowded_insert columns
## GPU
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert50seams_by_column_baseline.jpg" -dx 50 -dy 0 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert100seams_by_column_baseline.jpg" -dx 100 -dy 0 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_insert200seams_by_column_baseline.jpg" -dx 200 -dy 0 -checksum


## CPU
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert50seams_by_column_baseline.jpg" -dx 50 -dy 0
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert100seams_by_column_baseline.jpg" -dx 100 -dy 0
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_insert200seams_by_column_baseline.jpg" -dx 200 -dy 0

# crowded_remove rows
## GPU
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove50seams_by_row_baseline.jpg" -dx 0 -dy -50 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove100seams_by_row_baseline.jpg" -dx 0 -dy -100 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove200seams_by_row_baseline.jpg" -dx 0 -dy -200 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove200seams_by_row_and_column_baseline.jpg" -dx -200 -dy -200 -checksum

## CPU
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove50seams_by_row_baseline.jpg" -dx 0 -dy -50
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove100seams_by_row_baseline.jpg" -dx 0 -dy -100
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove200seams_by_row_baseline.jpg" -dx 0 -dy -200
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove200seams_by_row_and_column_baseline.jpg" -dx -200 -dy -200

# crowded_remove columns
## GPU
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove50seams_by_column_baseline.jpg" -dx -50 -dy 0 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove100seams_by_column_baseline.jpg" -dx -100 -dy 0 -checksum
python seam_carving_gpu.py -in "images/crowded.jpg" -out "ext_output_gpu/crowded_remove200seams_by_column_baseline.jpg" -dx -200 -dy 0 -checksum

## CPU
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove50seams_by_column_baseline.jpg" -dx -50 -dy 0
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove100seams_by_column_baseline.jpg" -dx -100 -dy 0
python seam_carving_cpu.py -in "images/crowded.jpg" -out "ext_output_cpu/crowded_remove200seams_by_column_baseline.jpg" -dx -200 -dy 0



# rain_origin_insert rows
Reference: https://www.verywellmind.com/an-overview-of-enochlophobia-4782189
## GPU
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert20seams_by_row_baseline.png" -dx 0 -dy 20 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert50seams_by_row_baseline.png" -dx 0 -dy 50 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert100seams_by_row_baseline.png" -dx 0 -dy 100 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert200seams_by_row_baseline.png" -dx 0 -dy 200 -checksu
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert200seams_by_row_and_column_baseline.png" -dx 200 -dy 200 -checksum

## CPU
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert20seams_by_row_baseline.png" -dx 0 -dy 20
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert50seams_by_row_baseline.png" -dx 0 -dy 50
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert100seams_by_row_baseline.png" -dx 0 -dy 100
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert200seams_by_row_baseline.png" -dx 0 -dy 200
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert200seams_by_row_and_column_baseline.png" -dx 200 -dy 200


# rain_origin_insert columns
## GPU
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert50seams_by_column_baseline.png" -dx 50 -dy 0 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert100seams_by_column_baseline.png" -dx 100 -dy 0 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_insert200seams_by_column_baseline.png" -dx 200 -dy 0 -checksum


## CPU
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert50seams_by_column_baseline.png" -dx 50 -dy 0
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert100seams_by_column_baseline.png" -dx 100 -dy 0
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_insert200seams_by_column_baseline.png" -dx 200 -dy 0

# rain_origin_remove rows
## GPU
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove50seams_by_row_baseline.png" -dx 0 -dy -50 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove100seams_by_row_baseline.png" -dx 0 -dy -100 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove200seams_by_row_baseline.png" -dx 0 -dy -200 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove200seams_by_row_and_column_baseline.png" -dx -200 -dy -200 -checksum

## CPU
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove50seams_by_row_baseline.png" -dx 0 -dy -50
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove100seams_by_row_baseline.png" -dx 0 -dy -100
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove200seams_by_row_baseline.png" -dx 0 -dy -200
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove200seams_by_row_and_column_baseline.png" -dx -200 -dy -200

# rain_origin_remove columns
## GPU
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove50seams_by_column_baseline.png" -dx -50 -dy 0 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove100seams_by_column_baseline.png" -dx -100 -dy 0 -checksum
python seam_carving_gpu.py -in "images/rain_origin.png" -out "ext_output_gpu/rain_origin_remove200seams_by_column_baseline.png" -dx -200 -dy 0 -checksum

## CPU
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove50seams_by_column_baseline.png" -dx -50 -dy 0
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove100seams_by_column_baseline.png" -dx -100 -dy 0
python seam_carving_cpu.py -in "images/rain_origin.png" -out "ext_output_cpu/rain_origin_remove200seams_by_column_baseline.png" -dx -200 -dy 0