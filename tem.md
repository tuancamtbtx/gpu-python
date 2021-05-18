| STT | Bước                                                        | Hình ảnh                                                                               |
| --- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 1   | Đọc ảnh RGB                                                 | ![](https://upload.wikimedia.org/wikipedia/commons/e/e2/BroadwayTowerSeamCarvingA.png) |
| 2   | Chuyển sang ảnh Grayscale                                   |                                                                                        |
| 3   | Tính độ quan trọng của mỗi pixel sử dụng gradient magnitude | ![](https://upload.wikimedia.org/wikipedia/commons/5/53/BroadwayTowerSeamCarvingB.png) |
| 4   | Từ bảng energy, tìm seam có độ quan trọng thấp nhất||
|5| Xóa các seams cho đến khi xóa đủ số seams cần xóa||
|