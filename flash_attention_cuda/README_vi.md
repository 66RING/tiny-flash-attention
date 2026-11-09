# Cài đặt flash attention bằng CUDA

LƯU Ý: cần phiên bản pytorch cụ thể để hỗ trợ các API đã bị loại bỏ (deprecated). Hãy dùng luôn phiên bản standalone hoặc phiên bản CUTLASS.

## Lộ trình

- [x] self attention đơn giản bằng python
- [x] self attention đơn giản bằng cuda
- [x] self attention đơn giản với python API binding
    - TODO:
        * hỗ trợ half (fp16)
        * làm kiểu dữ liệu template tổng quát hơn
        * cân bằng thread, và quá nhiều thread có thể gây crash
        * dọn các cảnh báo deprecated
- [x] flash attention 1 bằng cuda
- [x] flash attention 2 bằng cuda
- [x] python binding cho flash attention 1/2
- [ ] tách template và làm template tổng quát hơn (như dim và block size)
- [x] hỗ trợ MHA
- [ ] hỗ trợ chế độ causal
- [ ] flash attention với cute
- [x] tham khảo `static_switch.h` trong flash attention


## Kết quả

- Bạn cần **lập trình hướng kết quả (result-oriented programming)** trong CUDA
    * ví dụ: `C[x, y]` nên được tính bởi thread (x, y)
