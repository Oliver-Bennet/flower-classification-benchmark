# Flower Classification Benchmark

Framework PyTorch để huấn luyện và so sánh nhiều kiến trúc deep learning cho bài toán phân loại **Oxford Flowers 102**. Project hỗ trợ benchmark theo kiến trúc mô hình, optimizer, weight initialization và learning-rate scheduler.

## Tính năng

- Huấn luyện các mô hình `MLP`, `CNN`, `CNN + Transformer`, `ViT`, `Swin`, `ConvNeXt` và `MaxViT`.
- Cấu hình bằng YAML, không cần sửa code khi đổi experiment.
- Hỗ trợ Adam, AdamW, SGD, Cosine Annealing, StepLR và OneCycleLR.
- Data augmentation, ImageNet normalization, mixed precision và early stopping.
- Tự động lưu checkpoint tốt nhất, training history, summary JSON và biểu đồ.
- Đánh giá bằng accuracy, macro F1, weighted F1, classification report và confusion matrix.
- Có giao diện Streamlit để dự đoán ảnh và xem thông tin dataset/model.

## Cấu trúc thư mục

```text
flower-classification-benchmark/
├── app.py                    # Streamlit application
├── train.py                  # Entry point để huấn luyện
├── evaluate.py               # Đánh giá checkpoint trên test set
├── pyproject.toml            # Dependencies và thông tin project
├── uv.lock                   # Phiên bản dependency cố định
├── configs/
│   ├── config.yaml           # Cấu hình mặc định
│   └── experiments.yaml      # Các experiment E1-E15
├── data/flowers/             # Dataset local, không commit lên Git
│   ├── train/<class>/
│   ├── valid/<class>/
│   └── test/<class>/
├── datasets/                 # Dataset và transforms
├── models/                   # Model factory và các kiến trúc
├── engine/                   # Training, evaluation, inference, metrics
├── initialization/           # Weight initialization
├── optimizers/               # Optimizer factory
├── schedulers/               # Scheduler factory
├── utils/                    # Config, checkpoint, seed, visualization
└── outputs/                  # Checkpoint/log/result/figure, không commit
```

## Yêu cầu

- Windows, Linux hoặc macOS
- Python `3.13`
- GPU CUDA là tùy chọn; project tự chọn CUDA nếu khả dụng, nếu không sẽ dùng CPU
- [uv](https://docs.astral.sh/uv/) được khuyến nghị để cài dependency

## Cài đặt

Clone repository và đi vào thư mục project:

```bash
git clone <YOUR_REPOSITORY_URL>
cd flower-classification-benchmark
```

Cài môi trường và dependency bằng `uv`:

```bash
uv sync
```

Kích hoạt môi trường ảo nếu cần:

```powershell
.\.venv\Scripts\Activate.ps1
```

Hoặc chạy trực tiếp mọi lệnh qua `uv run` mà không cần kích hoạt môi trường:

```bash
uv run python train.py --experiment E2_cnn
```

## Chuẩn bị dữ liệu

Project sử dụng Flower Classification Dataset được cung cấp trên Kaggle:

[Flower Classification Dataset – Kaggle](https://www.kaggle.com/datasets/shahriar26s/flower-classification-dataset?utm_source=chatgpt.com)

Dataset phải có cấu trúc `ImageFolder` như sau:

```text
data/flowers/
├── train/
│   ├── 1/
│   │   ├── image_001.jpg
│   │   └── image_002.jpg
│   └── 2/
├── valid/
│   ├── 1/
│   └── 2/
└── test/
		├── 1/
		└── 2/
```

Mỗi thư mục con là một class. Tên và thứ tự class phải giống nhau giữa `train`, `valid` và `test`. Dataset được đọc bởi `torchvision.datasets.ImageFolder`.

Đổi đường dẫn dataset trong `configs/config.yaml`:

```yaml
data:
	root: "D:/flower-classification-benchmark/data/flowers"
```

Project mặc định dùng ảnh `224 x 224`, 102 class và normalization theo ImageNet.

## Huấn luyện

### Chạy một experiment

```bash
python train.py --experiment E2_cnn
```

Một số experiment tiêu biểu:

```bash
# So sánh kiến trúc
python train.py --experiment E1_mlp
python train.py --experiment E2_cnn
python train.py --experiment E3_cnn_transformer
python train.py --experiment E4_vit
python train.py --experiment E5_swin
python train.py --experiment E6_convnext
python train.py --experiment E7_maxvit

# So sánh optimizer
python train.py --experiment E8_sgd
python train.py --experiment E9_adam
python train.py --experiment E10_adamw

# So sánh weight initialization
python train.py --experiment E11_xavier
python train.py --experiment E12_kaiming

# So sánh scheduler
python train.py --experiment E13_steplr
python train.py --experiment E14_cosine
python train.py --experiment E15_onecycle
```

Có thể dùng tên đầy đủ theo nhóm trong YAML:

```bash
python train.py --experiment architecture.E4_vit
```

Đổi file config hoặc tên output:

```bash
python train.py \
	--config configs/config.yaml \
	--experiments configs/experiments.yaml \
	--experiment E2_cnn \
	--name cnn_trial_01
```

## Đánh giá checkpoint

Sau khi train, checkpoint tốt nhất thường nằm trong `outputs/checkpoints/`. Đánh giá trên test set bằng:

```bash
python evaluate.py \
	--checkpoint outputs/checkpoints/E2_cnn_best.pth \
	--experiment E2_cnn
```

Kết quả được lưu trong `outputs/results/`, còn confusion matrix được lưu trong `outputs/figures/`.

## Kết quả đầu ra

Mỗi lần train có thể tạo:

```text
outputs/
├── checkpoints/
│   └── <experiment>_best.pth
├── logs/
│   ├── <experiment>_history.json
│   └── <experiment>_history.csv
├── figures/
│   └── <experiment>_curves.png
└── results/
		├── <experiment>_summary.json
		└── <experiment>_test.json
```

Các file dataset, checkpoint và output đã được thêm vào `.gitignore` vì thường rất lớn và không nên commit lên GitHub.

## Chạy ứng dụng Streamlit

```bash
streamlit run app.py
```

Ứng dụng có các chức năng:

- upload ảnh hoa để dự đoán
- hiển thị top-k prediction
- xem gallery
- xem thông tin model
- xem thông tin dataset

Trước khi chạy app, cần bảo đảm các asset mà `app.py` tham chiếu tồn tại, gồm checkpoint, danh sách class, file thông tin hoa và metadata. Các đường dẫn này được khai báo ở phần `CONFIG` đầu file `app.py`.

## Luồng xử lý

```text
ImageFolder
		-> transforms
		-> DataLoader
		-> model
		-> CrossEntropyLoss
		-> backward + optimizer.step()
		-> validation
		-> checkpoint / metrics / plots
		-> test evaluation
```

`train.py` điều phối pipeline. `datasets/` chuẩn bị dữ liệu, `models/` xây dựng mạng, `engine/` xử lý train/evaluate, còn `utils/` phụ trách config, seed, checkpoint và visualization.

