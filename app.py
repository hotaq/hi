from pathlib import Path
import streamlit as st
from ultralytics import YOLO
from PIL import Image
def load_model(model_path):
   
    model = YOLO(model_path)
    return model

def infer_uploaded_image(conf, model):
   
   
    source_img = st.sidebar.file_uploader(
        label="Choose an image...",
        type=("jpg", "jpeg", "png", 'bmp', 'webp')
    )

    col1, col2 = st.columns(2)

    with col1:
        if source_img:
            uploaded_image = Image.open(source_img)
            st.image(
                image=source_img,
                caption="Uploaded Image",
                use_column_width=True
            )
    if source_img:
        if st.button("Execution"):
            with st.spinner("Running..."):
                res = model.predict(uploaded_image,
                                    conf=conf)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]

                with col2:
                    st.image(res_plotted,
                             caption="Detected Image",
                             use_column_width=True)
                    try:
                        with st.expander("Detection Results"):
                            for box in boxes:
                                st.write(box.xywh)
                    except Exception as ex:
                        st.write("No image is uploaded yet!")
                        st.write(ex)

st.set_page_config(
    page_title="object detection for detect any phase in mitosis cell",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("object detection for detect any phase in mitosis cell")

# sidebar
st.sidebar.header("Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        ['yolov8']
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100


if model_type:
    model_path = Path('yolov8.pt')
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: yolov8.pt")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    ['Image']
)

source_img = None
if source_selectbox == 'Image': # Image
    infer_uploaded_image(confidence, model)
