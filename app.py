"""
Streamlit demo for Flower Classification Benchmark.

Demo model: E3 – CNN + Transformer (best Accuracy / F1 trade-off).

Features:
  - Upload image or capture from camera → predict
  - Top-k predictions with confidence bars
  - Flower gallery (search + class list)
  - Model information + benchmark comparison table
  - Dataset information
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.transforms import build_eval_transforms
from models import build_model, count_parameters

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Flower Classification Demo",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths relative to project root (where you run `streamlit run app.py`)
ROOT = Path(__file__).resolve().parent

CHECKPOINT = ROOT / "outputs" / "checkpoints" / "E3_cnn_transformer_best.pth"
CLASS_JSON = ROOT / "metadata" / "class_names.json"
FLOWER_CSV = ROOT / "flower_info" / "flower_info.csv"
MODEL_INFO = ROOT / "metadata" / "model_info.json"
DATASET_INFO = ROOT / "metadata" / "dataset_info.json"
BENCHMARK_JSON = ROOT / "metadata" / "benchmark_summary.json"

# Optional: folder containing train/<class_name>/*.jpg for gallery previews
DATA_ROOT_CANDIDATES = [
    ROOT / "data" / "flowers",
    ROOT / "data" / "raw",
    Path(r"D:\flower-classification-benchmark\data\flowers"),
]

NUM_CLASSES = 102
IMAGE_SIZE = 224
TOP_K_DEFAULT = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Minimal config needed to rebuild E3 model + eval transforms
DEMO_CFG: Dict[str, Any] = {
    "data": {
        "image_size": IMAGE_SIZE,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "model": {
        "name": "cnn_transformer",
        "dropout": 0.3,
        "cnn_transformer": {
            "cnn_channels": [64, 128, 128],
            "embed_dim": 256,
            "num_heads": 4,
            "num_layers": 2,
            "mlp_ratio": 4.0,
            "dropout": 0.1,
        },
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_class_names() -> List[str]:
    """
    Class order MUST match training (ImageFolder sorts folder names
    lexicographically). Prefer metadata/class_names.json which was
    built in that order. Fallback: rebuild from numeric folders + CSV.
    """
    data = load_json(CLASS_JSON)
    if data is not None and len(data) == NUM_CLASSES:
        return list(data)

    # Rebuild from CSV + ImageFolder string sort of folder ids
    if FLOWER_CSV.is_file():
        df = pd.read_csv(FLOWER_CSV)
        id_to_name = dict(zip(df["id"].astype(int), df["name"]))
        folder_names = sorted(str(i) for i in range(NUM_CLASSES))
        return [id_to_name[int(f)] for f in folder_names]

    return [f"class_{i}" for i in range(NUM_CLASSES)]


def resolve_data_root() -> Optional[Path]:
    for p in DATA_ROOT_CANDIDATES:
        if p.is_dir():
            return p
    return None


def find_sample_image(class_id: int | str, data_root: Optional[Path]) -> Optional[Path]:
    """
    Tìm 1 ảnh ngẫu nhiên của lớp theo folder id (0, 1, 10, 100, ...).
    Thử lần lượt các split: train → test → valid → val.
    """
    if data_root is None:
        return None

    folder_name = str(int(class_id) + 1)          # thư mục thực tế là "0", "1", "10", ...
    candidates = []

    for split in ("train", "test", "valid", "val"):
        folder = data_root / split / folder_name
        if not folder.is_dir():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            candidates.extend(folder.glob(ext))

    if not candidates:
        return None

    return random.choice(candidates)


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> None:
    """Load state_dict from checkpoint (supports common key layouts)."""
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Please place E3_cnn_transformer_best.pth under outputs/checkpoints/"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    cleaned = {}
    for k, v in state.items():
        cleaned[k.replace("module.", "", 1) if k.startswith("module.") else k] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()


@st.cache_resource
def load_model_and_transform():
    """Load model once and cache it."""
    class_names = load_class_names()
    model = build_model(DEMO_CFG, num_classes=len(class_names))
    load_checkpoint_into_model(model, CHECKPOINT, DEVICE)
    model.to(DEVICE)
    transform = build_eval_transforms(DEMO_CFG)
    n_params = count_parameters(model)
    return model, transform, class_names, n_params


@torch.no_grad()
def predict_topk(
    model: torch.nn.Module,
    image: Image.Image,
    transform,
    class_names: List[str],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    model.eval()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
    logits = model(tensor)
    probs = F.softmax(logits, dim=1)[0]
    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k)
    return [(class_names[i], float(v)) for v, i in zip(values.tolist(), indices.tolist())]


def render_confidence_bars(predictions: List[Tuple[str, float]]) -> None:
    if not predictions:
        return
    df = pd.DataFrame(predictions, columns=["Flower", "Confidence"])
    df["Confidence %"] = (df["Confidence"] * 100).round(2)
    st.dataframe(
        df[["Flower", "Confidence %"]],
        use_container_width=True,
        hide_index=True,
    )
    for name, conf in predictions:
        st.write(f"**{name}** — {conf * 100:.2f}%")
        st.progress(min(max(conf, 0.0), 1.0))


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.title("🌸 Flower AI")
    st.caption("Benchmark Demo · CNN + Transformer")

    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔍 Predict", "🌼 Gallery", "📊 Model Info", "📁 Dataset"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Demo model**")
    st.markdown("E3 · CNN + Transformer")
    st.markdown("Accuracy **67.38%** · F1 **64.00%**")
    st.markdown(f"Device: `{DEVICE}`")

    if not CHECKPOINT.is_file():
        st.error(
            "Checkpoint missing!\n\n"
            f"Expected:\n`{CHECKPOINT.name}`\n\n"
            "Copy it to `outputs/checkpoints/`"
        )


# =============================================================================
# HOME
# =============================================================================

if page == "🏠 Home":
    st.title("🌸 Flower Classification Demo")
    st.markdown(
        """
        Ứng dụng minh họa kết quả đồ án  
        **Benchmarking Deep Learning Architectures and Training Strategies  
        for Flower Image Classification**.

        ### Chức năng chính
        | Trang | Mô tả |
        |-------|--------|
        | **Predict** | Upload ảnh hoặc chụp camera → nhận diện hoa, xem top-k |
        | **Gallery** | Duyệt 102 loài hoa trong dataset |
        | **Model Info** | Thông tin mô hình demo + bảng so sánh 7 kiến trúc |
        | **Dataset** | Thống kê bộ dữ liệu |

        ### Mô hình được chọn cho demo
        **E3 – CNN + Transformer** được chọn vì đạt Accuracy / Macro-F1 cao nhất  
        trong khi vẫn giữ số tham số và thời gian suy luận hợp lý (train from scratch).
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "67.38%")
    col2.metric("Macro F1", "64.00%")
    col3.metric("Parameters", "2.06M")
    col4.metric("Inference", "2.76 ms")

    st.info("➡️ Vào trang **Predict** để thử nhận diện ảnh hoa của bạn.")


# =============================================================================
# PREDICT
# =============================================================================

elif page == "🔍 Predict":
    st.title("🔍 Nhận diện hoa")
    st.markdown("Upload ảnh hoặc dùng camera, sau đó nhấn **Predict**.")

    try:
        model, transform, class_names, n_params = load_model_and_transform()
        model_ready = True
    except FileNotFoundError as e:
        st.error(str(e))
        model_ready = False
        model = transform = class_names = n_params = None
    except Exception as e:
        st.error(f"Không load được model: {e}")
        model_ready = False
        model = transform = class_names = n_params = None

    top_k = st.slider("Top-K predictions", min_value=1, max_value=10, value=TOP_K_DEFAULT)

    tab_upload, tab_camera = st.tabs(["📤 Upload ảnh", "📷 Chụp camera"])

    image: Optional[Image.Image] = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Chọn ảnh hoa (JPG / PNG)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded is not None:
            image = Image.open(uploaded).convert("RGB")

    with tab_camera:
        camera_photo = st.camera_input("Chụp ảnh hoa")
        if camera_photo is not None:
            image = Image.open(camera_photo).convert("RGB")

    if image is not None:
        col_img, col_pred = st.columns([1, 1.2])

        with col_img:
            st.subheader("Ảnh đầu vào")
            st.image(image, use_container_width=True)

        with col_pred:
            st.subheader("Kết quả dự đoán")
            if not model_ready:
                st.warning("Model chưa sẵn sàng. Kiểm tra checkpoint.")
            else:
                if st.button("🚀 Predict", type="primary", use_container_width=True):
                    with st.spinner("Đang suy luận..."):
                        preds = predict_topk(
                            model, image, transform, class_names, top_k=top_k
                        )
                    if preds:
                        best_name, best_conf = preds[0]
                        st.success(f"**{best_name}**  ·  {best_conf * 100:.2f}%")
                        st.markdown("---")
                        st.markdown(f"**Top-{top_k}**")
                        render_confidence_bars(preds)
                    else:
                        st.warning("Không có kết quả.")
    else:
        st.info("Hãy upload ảnh hoặc chụp camera để bắt đầu.")


# =============================================================================
# GALLERY
# =============================================================================

elif page == "🌼 Gallery":
    st.title("🌼 Flower Gallery")
    st.markdown("Danh sách 102 loài hoa trong dataset.")

    if not FLOWER_CSV.is_file():
        st.error(f"Không tìm thấy `{FLOWER_CSV}`")
    else:
        df = pd.read_csv(FLOWER_CSV).sort_values("id").reset_index(drop=True)
        data_root = resolve_data_root()
        if st.button("🔄 Random ảnh khác"):
            st.rerun()

        keyword = st.text_input("🔍 Tìm theo tên hoa", placeholder="ví dụ: rose, lily, daisy...")
        if keyword:
            mask = df["name"].str.contains(keyword, case=False, na=False)
            df_view = df[mask]
        else:
            df_view = df

        st.caption(f"Hiển thị {len(df_view)} / {len(df)} loài")

        n_cols = 4
        rows = (len(df_view) + n_cols - 1) // n_cols
        for r in range(rows):
            cols = st.columns(n_cols)
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx >= len(df_view):
                    break
                row = df_view.iloc[idx]
                with cols[c]:
                    sample = find_sample_image(str(row["id"]), data_root) if data_root else None
                    if sample is not None:
                        try:
                            st.image(str(sample), use_container_width=True)
                        except Exception:
                            st.markdown("🖼️")
                    else:
                        st.markdown(
                            "<div style='height:120px;background:#f0f0f0;"
                            "border-radius:8px;display:flex;align-items:center;"
                            "justify-content:center;color:#888;'>No preview</div>",
                            unsafe_allow_html=True,
                        )
                    st.markdown(f"**{int(row['id']) + 1}. {row['name']}**")


# =============================================================================
# MODEL INFO
# =============================================================================

elif page == "📊 Model Info":
    st.title("📊 Model Information")

    info = load_json(MODEL_INFO)
    if info:
        st.subheader("Mô hình demo (E3)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Architecture", info.get("architecture", "—"))
        c2.metric("Parameters", info.get("parameters", "—"))
        c3.metric("Inference", info.get("inference_ms", "—"))

        c4, c5, c6 = st.columns(3)
        c4.metric("Accuracy", info.get("accuracy", "—"))
        c5.metric("Macro F1", info.get("macro_f1", "—"))
        c6.metric("Classes", info.get("num_classes", "—"))

        c7, c8, c9 = st.columns(3)
        c7.metric("Precision", info.get("precision", "—"))
        c8.metric("Recall", info.get("recall", "—"))
        c9.metric("Training Time", f"{info.get('training_time_min', '—')} min")

        st.markdown("**Mô tả**")
        st.write(info.get("description", ""))
        st.markdown(
            f"- Optimizer: `{info.get('optimizer')}`  \n"
            f"- Scheduler: `{info.get('scheduler')}`  \n"
            f"- Initialization: `{info.get('initialization')}`  \n"
            f"- Train from scratch: `{info.get('trained_from_scratch')}`  \n"
            f"- Checkpoint: `{info.get('checkpoint')}`"
        )
        if info.get("notes"):
            st.info(info["notes"])
    else:
        st.warning("Không tìm thấy metadata/model_info.json")

    st.divider()
    st.subheader("So sánh 7 kiến trúc (Architecture Benchmark)")

    bench = load_json(BENCHMARK_JSON)
    if bench and "architectures" in bench:
        rows = []
        for a in bench["architectures"]:
            rows.append(
                {
                    "ID": a["id"],
                    "Model": a["name"],
                    "Accuracy (%)": a["accuracy"],
                    "Precision (%)": a.get("precision", "—"),
                    "Recall (%)": a.get("recall", "—"),
                    "Macro F1 (%)": a["macro_f1"],
                    "Params (M)": a["params_m"],
                    "Train (min)": a.get("training_time_min", "—"),
                    "Inference (ms)": a["inference_ms"],
                    "Demo": "✅" if a.get("selected_for_demo") else "",
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if bench.get("note"):
            st.caption(bench["note"])
    else:
        st.markdown(
            """
            | Model                | Accuracy | Macro F1 | Params | Inference / ảnh |
            | -------------------- | -------- | -------- | ------ | --------------- |
            | E1 MLP               | 33.45%   | 28.30%   | 0.55M  | 13.90 ms        |
            | E2 CNN               | 51.57%   | 45.51%   | 0.42M  | 2.99 ms         |
            | **E3 CNN + Transformer** | **67.38%** | **64.00%** | **2.06M** | **2.76 ms** |
            | E4 ViT               | 37.02%   | 31.72%   | 11.06M | 2.42 ms         |
            | E5 Swin              | 4.95%    | 1.15%    | 27.60M | 5.93 ms         |
            | E6 ConvNeXt          | 49.07%   | 45.81%   | 27.90M | 10.78 ms        |
            | E7 MaxViT            | 58.52%   | 54.66%   | 30.46M | 13.88 ms        |
            """
        )


# =============================================================================
# DATASET
# =============================================================================

elif page == "📁 Dataset":
    st.title("📁 Dataset Information")

    info = load_json(DATASET_INFO)
    if info:
        c1, c2, c3, c4 = st.columns(4)
        total = info.get("total_images", "—")
        c1.metric("Total images", f"{total:,}" if isinstance(total, int) else total)
        c2.metric("Classes", info.get("num_classes", "—"))
        c3.metric("Image size", f"{info.get('image_size')}×{info.get('image_size')}")
        c4.metric("Split", info.get("split_ratio", "—"))

        c5, c6, c7 = st.columns(3)
        c5.metric("Train", info.get("train", "—"))
        c6.metric("Validation", info.get("validation", "—"))
        c7.metric("Test", info.get("test", "—"))

        st.markdown(
            f"""
            - **Name:** {info.get('name')}  
            - **Source:** [{info.get('source', '')}]({info.get('source', '')})  
            - **Preprocessing:** {info.get('preprocessing')}  
            - **License:** {info.get('license')}  
            - **Task:** {info.get('task')}
            """
        )
    else:
        st.warning("Không tìm thấy metadata/dataset_info.json")

    st.divider()
    st.subheader("Danh sách lớp (102 flowers)")
    if FLOWER_CSV.is_file():
        df = pd.read_csv(FLOWER_CSV).sort_values("id")
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)
    else:
        names = load_class_names()
        st.dataframe(
            pd.DataFrame({"id": range(len(names)), "name": names}),
            use_container_width=True,
            hide_index=True,
            height=400,
        )