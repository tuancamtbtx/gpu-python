# ĐỒ ÁN CUỐI KÌ GPU

## Cài đặt seam carving

## Thành viên:
| No. | Họ và tên       | MSSV    | Account Github                                        | Email                         |
| --- | --------------- | ------- | ----------------------------------------------------- | ----------------------------- |
| 1   | Nguyễn Văn Tuấn | 1512638 | [tuancamtbtx](https://github.com/tuancamtbtx)         | nguyenvantuan140397@gmail.com |
| 2   | Lê Thanh Hiếu   | 1712434 | [hieulethanh-dev](https://github.com/hieulethanh-dev) | hieulethanh.dev@gmail.com     |
| 3   | Phan Hữu Tú     | 1712861 | [tuhp-tech](https://github.com/tuhp-tech)             | tuhp.tech@gmail.com           |

## Phân công công việc
- Tuần 01:
  - Tìm hiểu thuật toán, numba nopython
    - Thành viên tham gia: Cả nhóm
  - Cài đặt tuần tự (dự kiến 90%):
    - Thành viên tham gia: Cả nhóm
  - Chuẩn bị báo cáo, viết notebook và demo:
    - Thành viên tham gia: Hiếu
- Tuần 02:
  - Giải quyết các yêu cầu, góp ý của thầy và các nhóm:
    - Bổ sung file README, Notebook: phân công công việc chi tiết hơn, bổ sung thông tin thuật toán chèn thêm seam, bổ sung phần mô tả bài toán
      - Thành viên tham gia: Tuấn, Hiếu
    - Giải quyết câu hỏi, góp ý của các nhóm: về chiến lược thêm, xóa ảnh; xóa theo chiều ngang có tốt hơn?
      - Thành viên tham gia: Tú
  - Tìm cách tối ưu hóa công thức tính energy:
    - Thành viên tham gia: Tú
  - Chuẩn bị báo cáo, notebook, demo cho buổi báo cáo tuần 03:
    - Thành viên tham gia: Tuấn, Tú
- Tuần 03:
  - Tìm hiểu về song song hóa trên numba
    - Thành viên tham gia: Cả nhóm
    - Deadline: 09/05
  - Họp nhóm và phân chia công việc song song hóa
    - Thành viên: Cả nhóm
    - Thời gian: 09/05
  - Cài đặt baseline song song:
    - Thành viên tham gia: Cả nhóm
- Tuần 04:
  - Viết lại notebook: Cả nhóm 
  - Hoàn thành cài đặt tuần tự: Cả nhóm
  - Chuẩn bị nội dung báo cáo hàng tuần: Tuấn, Hiếu
- Tuần 05:
- Tuần 06:
- Tuần 07:
- Tuần 08:

## Cách tổ chức:
### Tham số dòng lệnh:
- seam_carving.py [-mode <running_mode>] [-dx <seam_dx>] [-dy <seam_dy>] [-in <image_in>] [-out <image_out>]
- Với:
  + mode: cpu hoặc gpu
  + dx: số pixels cần thay đổi theo chiều ngang của ảnh
  + dy: số pixels cần thay đổi theo chiều dọc của ảnh
  + image_in: ảnh gốc được đưa vào
  + image_out: ảnh sau khi sử dụng seam carving


#### 3. Tìm Seam có chi phí nhỏ nhất
- Quay lui từ hàng cuối cùng của ma trận lên hàng thứ 0 để tìm seam có chi phí nhỏ nhất
![](/readmeimages/algo3.png)

#### 4. Xóa Seam nhỏ nhất tìm được ở bước 3 
- Khởi tạo 1 ma trận boolean có kích thước tương tự ma trận energy toàn bộ phần tử có giá trị True làm mask
#### 5. Lặp lại các bước từ 1 tới 4 cho đến khi đủ số lượng seams cần xóa

### Thuật toán chèn thêm seams
- Tương tự như thuật toán xóa seams nhưng ngược lại thay vì xóa seams nhỏ nhất ta sẽ nhân bản chúng.
  - Đầu tiên ta thực hiện xóa n seams trên 1 images duplicate từ image gốc và ghi lại theo đúng thứ tự tọa độ các seams bị xóa. Sau đó chèn các seams vào ảnh gốc theo thứ tự chúng được xóa


## Quy trình tối ưu hóa
### Bảng đo thời gian

## Quy trình tối ưu với GPU (dự kiến)
- Tính Energy map bằng kernel function
- Tính Minimum Cost Table bằng kernel function
- Xác định seam cần xóa bằng kernel function
- Sử dụng shared memory

## Tham khảo
- https://github.com/kalpeshdusane/Seam-Carving-B.E.-Project
- Slides đồ án môn Lập trình song song
- https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
- https://andrewdcampbell.github.io/seam-carving
