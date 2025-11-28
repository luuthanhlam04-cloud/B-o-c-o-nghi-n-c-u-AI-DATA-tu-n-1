# Nghiên cứu tìm hiểu về AI - Tuần 1
## I. Giới thiệu về AI

### 1. Khái niệm và Phân loại
* **Định nghĩa:** AI là tập hợp các phương pháp, thuật toán và hệ thống giúp máy tính thực hiện các nhiệm vụ cần trí tuệ con người như nhận diện, suy luận, học từ dữ liệu và ra quyết định.
* **Kỹ thuật:** Bao gồm mô hình toán học, thống kê và tối ưu hóa để tìm quy luật từ dữ liệu hoặc môi trường.

### 2. Các lĩnh vực chính
* **Machine Learning (ML):** Thuật toán học từ dữ liệu để dự đoán/phân loại thay vì lập trình thủ công. Gồm: Học có giám sát, Học không giám sát, Học bán giám sát, Học tăng cường.
* **Deep Learning (DL):** Tập con của ML dùng mạng nơ-ron nhiều lớp để học biểu diễn phức tạp từ dữ liệu thô (ảnh, âm thanh).
* **Xử lý ngôn ngữ tự nhiên (NLP):** Phân tích cú pháp, cảm xúc, dịch máy, tóm tắt văn bản.
* **Thị giác máy tính (Computer Vision):** Phân loại, phát hiện, phân đoạn ảnh, tái tạo 3D.
* **Robotics:** Kết hợp nhận thức, lập kế hoạch và điều khiển để robot tương tác với thế giới thực.
* **Hệ chuyên gia (Knowledge-based systems):** Suy luận dựa trên luật hoặc biểu diễn tri thức.

### 3. Ứng dụng thực tế
* **Nhận diện khuôn mặt:** Dùng DL để phân biệt người.
* **Chatbot/Trợ lý ảo:** Dùng NLP để quản lý hội thoại và thực hiện tác vụ.
* **Hệ gợi ý:** Dùng lịch sử người dùng để dự đoán sản phẩm phù hợp.
* **Phát hiện gian lận:** Dùng Anomaly detection để tìm hành vi bất thường.
* **Xe tự lái:** Tích hợp CV, định vị, lập kế hoạch và điều khiển.
* **Y tế:** Phân tích ảnh y khoa phát hiện tổn thương.
* **Tự động hóa sản xuất:** Robot nhận diện lỗi, tối ưu năng suất.

## II. Dữ liệu (Data)

### 1. Định nghĩa và Phân loại
* **Định nghĩa:** Tập các quan sát/bản ghi (số, văn bản, ảnh, tín hiệu...).
* **Cấu trúc:**
    * Structured: Bảng, CSDL, cột rõ ràng.
    * Unstructured: Văn bản, ảnh, video.
    * Semi-structured: JSON, XML.

### 2. Vai trò của dữ liệu trong AI
1.  **Nguồn học:** Mô hình tối ưu tham số dựa trên dữ liệu huấn luyện.
2.  **Chất lượng:** Dữ liệu sai lệch dẫn đến mô hình sai (Garbage in - Garbage out).
3.  **Số lượng & Đa dạng:** Cần lượng lớn dữ liệu để học biểu diễn phức tạp và giảm overfitting.
4.  **Nhãn (Label):** Nhãn chất lượng cao quyết định độ chính xác của học có giám sát.
5.  **Cân bằng & Thiên lệch:** Dữ liệu mất cân bằng (imbalance) gây thiên vị cho mô hình.
6.  **Tiền xử lý:** Cần làm sạch, chuẩn hóa, trích xuất đặc trưng.
7.  **Chia tách:** Tách tập train/validation/test để đánh giá hiệu suất.
8.  **Concept Drift:** Cần cập nhật dữ liệu mới khi môi trường thay đổi.
9.  **Pháp lý & Chi phí:** Tuân thủ quyền riêng tư (GDPR) và tối ưu chi phí thu thập/lưu trữ.

## III. Phân loại bài toán Machine Learning

| Tiêu chí | Học có giám sát (Supervised) | Học không giám sát (Unsupervised) | Học tăng cường (Reinforcement) |
| :--- | :--- | :--- | :--- |
| **Dữ liệu** | Có nhãn $(X, y)$ | Không nhãn (chỉ $X$) | Trạng thái từ môi trường |
| **Mục tiêu** | Dự đoán output từ input | Tìm cấu trúc/mẫu ẩn | Tối đa hóa phần thưởng |
| **Cách học** | Tối ưu sai số so với nhãn thật | Phân tích phân bố/mật độ | Thử - sai, phản hồi reward |
| **Đầu ra** | Phân loại, Hồi quy | Cụm, luật kết hợp | Chính sách hành động |
| **Ví dụ** | Dự đoán giá nhà, spam | Phân cụm khách hàng | Chơi game, robot |

## IV. Thuật ngữ và Ký hiệu

* **Dữ liệu:** Dataset, Sample, Feature ($x$), Label ($y$), Training/Test set.
* **Mô hình:** Model $f(x)$, Parameters $\theta$, Architecture, Capacity.
* **Huấn luyện:** Loss function $L$, Gradient Descent, Learning rate $\alpha$, Epoch, Batch, Overfitting/Underfitting.
* **Supervised:** Classification, Regression, Metrics (Accuracy, Precision, Recall, F1, MSE).
* **Unsupervised:** Clustering, Centroid, Dimensionality Reduction (PCA).
* **Deep Learning:** Neuron, Layer, Activation function, Backpropagation, CNN, RNN, Transformer.
* **Reinforcement Learning:** Agent, Environment, State ($s$), Action ($a$), Reward ($r$), Policy ($\pi$), Q-value.
* **Toán học:** Expectation, Variance, Likelihood, Bias-Variance Tradeoff.

## V. Các thư viện phổ biến

1.  **Numpy:**
    * Tính toán số học, đại số tuyến tính, mảng/ma trận hiệu quả.
    * Nền tảng cho Pandas, PyTorch.
2.  **Pandas:**
    * Xử lý dữ liệu dạng bảng (DataFrame), lọc, nhóm, phân tích.
    * Hỗ trợ nhiều nguồn (CSV, Excel, SQL).
3.  **Matplotlib:**
    * Vẽ biểu đồ 2D (line, scatter, bar).
    * Tùy chỉnh linh hoạt nhưng cú pháp phức tạp.
4.  **PyTorch:**
    * Thư viện Deep Learning, hỗ trợ GPU, tính toán gradient tự động.
    * Dễ học, cú pháp giống Python.

## VI. Nguồn dữ liệu và Nền tảng

### 1. Phân loại nguồn dữ liệu
* Nguồn mở (Open Data), Cộng đồng, API công khai, Doanh nghiệp.
* Thu thập thủ công: Web scraping, khảo sát, IoT.

### 2. Nền tảng nổi bật
* **Kaggle:** Kho dữ liệu lớn và cuộc thi ML. Các dataset: Titanic, House Prices, MNIST, CIFAR-10, Netflix.
* **Hugging Face:** Nền tảng mã nguồn mở cho NLP/Multimodal. Cung cấp Model Hub, Dataset Hub, Transformers Library.

## VII. Phân tích dữ liệu sơ bộ (EDA)

**Khái niệm:** Phương pháp tiếp cận để hiểu bản chất dữ liệu trước khi mô hình hóa.

### 1. Hiểu dữ liệu (Data Understanding)
* Kiểm tra cấu trúc (Shape), loại dữ liệu (Types).
* Đánh giá chất lượng: Giá trị thiếu (Missing), Trùng lặp (Duplicates), Tính nhất quán.

### 2. Phân tích dữ liệu (Data Analysis)
* Xu hướng tập trung: Mean, Median, Mode.
* Độ phân tán: Variance, Std Dev, Range.
* Phân phối: Skewness, Kurtosis.
* Tương quan: Hệ số Pearson, Spearman.

### 3. Trực quan hóa dữ liệu (Data Visualization)
* **Đơn biến:** Histogram (phân phối), Boxplot (outliers), Bar chart.
* **Đa biến:** Scatter Plot (quan hệ 2 biến), Heatmap (tương quan), Pair Plot.
