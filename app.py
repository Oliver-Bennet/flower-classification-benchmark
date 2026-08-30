import json

import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from PIL import Image

from datasets.flower_dataset import get_transforms
from models.cnn import SimpleCNN
from utils.inference import Predictor

# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Flower Classification",
    page_icon="🌸",
    layout="wide"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT = "outputs/checkpoints/best_model.pth"
CLASS_JSON = "metadata/class_names.json"
FLOWER_CSV = "flower_info/flower_info.csv"
MODEL_INFO = "metadata/model_info.json"
DATASET_INFO = "metadata/dataset_info.json"

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_predictor():

    _, transform = get_transforms()

    model = SimpleCNN(
        num_classes=102
    )

    predictor = Predictor(
        model=model,
        checkpoint=CHECKPOINT,
        transform=transform,
        csv_path=FLOWER_CSV,
        device=DEVICE
    )

    return predictor


predictor = load_predictor()

flower_df = pd.read_csv(FLOWER_CSV)

flower_df["id"] = flower_df.id + 1
# ==========================================================
# SIDEBAR
# ==========================================================

from streamlit_option_menu import option_menu

with st.sidebar:

    page = option_menu(
        "Flower AI",
        [
            "Home",
            "Predict",
            "Gallery",
            "About Model",
            "About Dataset"
        ],
        icons=[
            "house",
            "search",
            "images",
            "cpu",
            "database"
        ],
        default_index=0
    )

# page = st.sidebar.radio(

#     "Navigation",

#     [
#         "🏠 Home",
#         "🔍 Predict",
#         "🌼 Flower Gallery",
#         "📊 About Model",
#         "📁 About Dataset"
#     ]
# )

# ==========================================================
# HOME
# ==========================================================
import random
from pathlib import Path

DATASET = Path("D:\\flower-classification-benchmark\\data\\flowers\\train")

def get_random_image(class_id):

    class_id = int(class_id)      # ép về int

    folder = DATASET / str(class_id)

    if not folder.exists():
        st.write("Không tồn tại:", folder)
        return None

    imgs = [
        p for p in folder.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ]

    if not imgs:
        st.write("Không có ảnh:", folder)
        return None

    return random.choice(imgs)

if page == "Home":

    st.title("🌸 Flower Classification AI")

    st.write(
        """
        Deep Learning application for recognizing flower species.

        This project benchmarks multiple deep learning architectures
        and deploys the best model using Streamlit.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset",
            "Oxford Flowers 102"
        )

    with col2:
        st.metric(
            "Classes",
            "102"
        )

    with col3:
        st.metric(
            "Model",
            "ConvNeXt Tiny"
        )

    st.divider()

    st.subheader("Project Features")

    st.markdown(
        """
        ✅ Upload flower image

        ✅ Top-5 prediction

        ✅ Flower information

        ✅ Flower gallery

        ✅ Model information

        ✅ Dataset statistics
        """
    )

# ==========================================================
# PREDICT
# ==========================================================

elif page == "Predict":

    st.title("🔍 Flower Prediction")

    uploaded = st.file_uploader(

        "Upload an image",

        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            result = predictor.predict(image)

            top5 = predictor.predict_topk(image)

            info = predictor.get_flower_info(
                result["id"]
            )

            st.success(result["name"])

            st.metric(

                "Confidence",

                f"{result['confidence']*100:.2f}%"
            )

            st.subheader("Top-5 Predictions")

            df = pd.DataFrame(top5)

            fig = px.bar(

                df,

                x="confidence",

                y="name",

                orientation="h",

                text="confidence"
            )

            fig.update_traces(
                texttemplate="%{text:.2%}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.divider()

        st.header("Flower Information")

        cols = st.columns(2)

        keys = [

            "scientific_name",

            "family",

            "color",

            "habitat"
        ]

        for c, k in zip(cols * 2, keys):

            if k in info:

                c.write(f"**{k.replace('_',' ').title()}**")

                c.write(info[k])

        if "description" in info:

            st.write("### Description")

            st.write(info["description"])

# ==========================================================
# GALLERY
# ==========================================================



elif page == "Gallery":

    st.title("🌼 Flower Gallery")

    keyword = st.text_input(
        "Search flower"
    )

    df = flower_df

    if keyword:

        df = df[
            df["name"].str.contains(
                keyword,
                case=False
            )
        ]

    cols = st.columns(4)

    for idx, row in enumerate(df.itertuples()):

        with cols[idx % 4]:

            preview = get_random_image(row.id)

            if preview is not None:
                st.image(str(preview), use_container_width=True)

            st.write(f"{row.id} - {row.name}")

            if st.button(
                f"View {row.id}"
            ):

                st.session_state["flower"] = row.id

    if "flower" in st.session_state:

        fid = st.session_state["flower"]

        info = predictor.get_flower_info(fid)

        st.divider()

        st.header(info["name"])

        preview = get_random_image(fid)

        if preview:
            st.image(str(preview), width=350)

        st.subheader("Flower Information")

        for k, v in info.items():

            st.write(f"**{k}** : {v}")

# ==========================================================
# ABOUT MODEL
# ==========================================================

elif page == "About Model":

    st.title("📊 Model Information")

    try:

        with open(MODEL_INFO) as f:

            model = json.load(f)

        col1, col2 = st.columns(2)

        for i, (k, v) in enumerate(model.items()):

            if i % 2 == 0:

                col1.metric(k, v)

            else:

                col2.metric(k, v)

    except:

        st.warning(
            "model_info.json not found."
        )

# ==========================================================
# ABOUT DATASET
# ==========================================================

elif page == "About Dataset":

    st.title("📁 Dataset")

    try:

        with open(DATASET_INFO) as f:

            data = json.load(f)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Train",
            data["train"]
        )

        col2.metric(
            "Validation",
            data["validation"]
        )

        col3.metric(
            "Test",
            data["test"]
        )

        st.metric(
            "Classes",
            data["classes"]
        )

        st.metric(
            "Image Size",
            f"{data['image_size']} x {data['image_size']}"
        )

    except:

        st.warning(
            "dataset_info.json not found."
        )