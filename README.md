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
- Tuần 04:
  - Viết lại notebook: Cả nhóm 
  - Hoàn thành cài đặt tuần tự: Cả nhóm
  - Chuẩn bị nội dung báo cáo hàng tuần: Tuấn, Hiếu
- Tuần 05:
  - Cài đặt baseline song song:
    - Thành viên tham gia: Cả nhóm
- Tuần 06:
  - Song song hóa bước chuyển đổi ảnh màu thành ảnh xám.
    - Thành viên: Tú
  - Song song hóa bước tìm biên cạnh cho ảnh xám.
    - Thành viên: Tuấn
  - Cài đặt song song hóa hàm forward energy.
    - Thành viên: Hiếu
- Tuần 07:
  - Bổ sung báo cáo: thêm đánh giá (ảnh) giữa orginal paper và kết quả cài đặt của nhóm cho forward energy,...
    - Thành viên: Tuấn
  - Cài đặt song song hóa hàm forward energy
    - Thành viên: Hiếu
  - Chuẩn bị nội dung báo cáo hàng tuần
    - Thành viên: Hiếu
  - Cài đặt song song hóa bước xóa seam, chèn seam:
    - Thành viên: Tú
- Tuần 08:
  - Chuẩn bị nội dung báo cáo hàng tuần
    - Thành viên: Tú
  - Sắp xếp và kết hợp các hàm đã song song hóa để chạy toàn bộ chương trình
    - Thành viên: Hiếu
  - Hoàn thành cài đặt song song hóa baseline: Cả nhóm
  - Đưa lên Colab để chạy thực nghiệm:
    - Thành viên: Tuấn
- Tuần 09:
  - Hoàn thành cài đặt song song hóa phiên bản v1: Cả nhóm
  - Bổ sung báo cáo: đánh giá (ảnh) giữa cpu và gpu numba implementation
    - Thành viên: Hiếu
  - Cải tiến song song hóa (hạn chế in-out bound ở bộ nhớ, cache ở cuda):
    - Thành viên: Tú
  - Cải tiến song song hóa (tính toán kernel, chạy thực nghiệm để đánh giá): Cả nhóm
  - Hoàn thành cài đặt cải tiến song song hóa phiên bản v2: Cả nhóm
- Tuần 10:
  - Bổ sung báo cáo: bổ sung đánh giá (ảnh) giữa cpu và gpu numba implementation
    - Thành viên: Tuấn
  - Đóng góp và sửa chữa báo cáo cho hoàn thiên: Cả nhóm
## Cách tổ chức:
### Tham số dòng lệnh:
- seam_carving.py [-mode <running_mode>] [-dx <seam_dx>] [-dy <seam_dy>] [-in <image_in>] [-out <image_out>]
- Với:
  + mode: cpu hoặc gpu
  + dx: số pixels cần thay đổi theo chiều ngang của ảnh
  + dy: số pixels cần thay đổi theo chiều dọc của ảnh
  + in: ảnh gốc được đưa vào
  + out: ảnh sau khi sử dụng seam carving (path pattern cpu: output_cpu, gpu: output_gpu)

- seam_carving_cpu.py [-dx <seam_dx>] [-dy <seam_dy>] [-in <image_in>] [-out <image_out>] -test_time
- Với:
  + mode: cpu hoặc gpu
  + dx: số pixels cần thay đổi theo chiều ngang của ảnh
  + dy: số pixels cần thay đổi theo chiều dọc của ảnh
  + in: ảnh gốc được đưa vào
  + out: ảnh sau khi sử dụng seam carving (path pattern cpu: output_cpu, gpu: output_gpu)
  + test_time: in ra thời gian chạy của từng hàm trong pipeline

- Điều chỉnh BLOCK_SIZE trong gpu
  + export BLOCK_SIZE_1D=32
  + export BLOCK_SIZE_2D=32
- seam_carving_gpu.py [-dx <seam_dx>] [-dy <seam_dy>] [-in <image_in>] [-out <image_out>] -check_sum
- Với:
  + mode: cpu hoặc gpu
  + dx: số pixels cần thay đổi theo chiều ngang của ảnh
  + dy: số pixels cần thay đổi theo chiều dọc của ảnh
  + in: ảnh gốc được đưa vào
  + out: ảnh sau khi sử dụng seam carving (path pattern cpu: output_cpu, gpu: output_gpu)
  + check_sum: sử dụng grayscale của cpu (tránh sai số)


## Tham khảo
- https://github.com/kalpeshdusane/Seam-Carving-B.E.-Project
- Slides đồ án môn Lập trình song song
- https://avikdas.com/2019/07/29/improved-seam-carving-with-forward-energy.html
- https://andrewdcampbell.github.io/seam-carving
- https://numba.pydata.org/numba-doc/latest/cuda/index.html

## Input/Output thực nghiệm các hình khác
- https://drive.google.com/drive/folders/1kIN9H2TTnlZ2BWiorMmnpONIN0pKyeTb?usp=sharing