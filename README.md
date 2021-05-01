# ĐỒ ÁN CUỐI KÌ GPU

## Cài đặt seam carving

## Thành viên:
| No.  | Họ và tên | MSSV | Email |
| ------------- | ------------- | ------------- | ------------- |
| 1  | Nguyễn Văn Tuấn | 1512638 | nguyenvantuan140397@gmail.com|
| 2  | Lê Thanh Hiếu | 1712434 | hieulethanh.dev@gmail.com| 
| 3  | Phan Hữu Tú  | 1712861| tuhp.tech@gmail.com |


## Cách tổ chức:
### Tham số dòng lệnh:
- seam_carving.py [-mode <running_mode>] [-dx <seam_dx>] [-dy <seam_dy>] [-in <image_in>] [-out <image_out>]
- Với:
  + mode: cpu hoặc gpu
  + dx: chiều ngang mới của ảnh
  + dy: chiều dọc mới của ảnh
  + image_in: ảnh gốc được đưa vào
  + image_out: ảnh sau khi sử dụng seam carving

### Ví dụ:
- python seam_carving.py -mode cpu -dx 10 -dy 0 -in /home/tuhp/study/LTSSUD/gpu-python/pietro_first_seam.jpg -out /home/tuhp/study/LTSSUD/gpu-python/pietro_first_seam_out.jpg

### Các tính năng cần thực hiện:

- Chỉnh kích thước ảnh theo chiều ngang của ảnh được đưa vào
- Chỉnh kích thước ảnh theo chiều dọc của ảnh được đưa vào
