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
  + dx: số pixels cần thay đổi theo chiều ngang của ảnh
  + dy: số pixels cần thay đổi theo chiều dọc của ảnh
  + image_in: ảnh gốc được đưa vào
  + image_out: ảnh sau khi sử dụng seam carving

## Chi tiết thuật toán
- Các tính năng cần thực hiện:
  - Xóa seams (giảm kích thước của ảnh) theo chiều cao, rộng của ảnh
  - Thêm seams (tăng kích thước của ảnh) theo chiều cao, rộng của ảnh

### Thuật toán xóa seams(tuần tự)
#### 1. Tính Energy Map 
#### 2. Tìm bảng chi phí nhỏ nhất (Minimum Cost Table)
#### 3. Tìm Seam có chi phí nhỏ nhất
#### 4. Xóa Seam nhỏ nhất tìm được ở bước 3 
#### 5. Lặp lại các bước từ 1 tới 4 cho đến khi đủ số lượng seams cần xóa

### Thuật toán chèn thêm seams
- Tương tự như thuật toán xóa seams nhưng ngược lại thay vì xóa seams nhỏ nhất ta sẽ nhân bản chúng.
  - Đầu tiên ta thực hiện xóa n seams trên 1 images duplicate từ image gốc và ghi lại theo đúng thứ tự tọa độ các seams bị xóa. Sau đó chèn các seams vào ảnh gốc theo thứ tự chúng được xóa

## Tối ưu với GPU (dự kiến)
- Tính Energy map bằng kernel function
- Tính Minimum Cost Table bằng kernel function
- Sử dụng shared memory

## Tham khảo
- https://github.com/kalpeshdusane/Seam-Carving-B.E.-Project