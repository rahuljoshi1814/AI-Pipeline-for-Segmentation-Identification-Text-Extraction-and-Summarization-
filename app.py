import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import modules
from models.loaders import load_maskrcnn, load_clip_model, load_easyocr_reader
from models.segmentation import segment_image
from models.identification import identify_objects
from models.text_extraction import extract_text
from utils.visualization import draw_annotations
from utils.results import build_results

# Device
device = "cuda" if st.sidebar.checkbox("Use CUDA if available", value=True) and "cuda" in str(next(iter(load_maskrcnn().parameters())).device) else "cpu"

# Load models
maskrcnn = load_maskrcnn()
clip_model, clip_preprocess = load_clip_model()
ocr_reader = load_easyocr_reader()

# Candidate labels
candidate_labels = [
    "person", "dog", "cat", "laptop", "bicycle", "car",
    "book", "bottle", "cup", "phone", "chair", "table",
    "plant", "keyboard", "monitor", "pen", "bag"
]

st.title("🔍 Vision AI: Image Segmentation, CLIP Labeling & OCR")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Segmenting objects..."):
        objects = segment_image(image_np, maskrcnn)

    with st.spinner("Labeling objects with CLIP..."):
        objects = identify_objects(objects, clip_model, clip_preprocess, candidate_labels, device=device)

    with st.spinner("Extracting text (OCR)..."):
        text_data = extract_text(image_np, ocr_reader)

    # Build combined results
    results = build_results(uploaded_file.name, objects, text_data)

    # Draw annotated image
    annotated = draw_annotations(image_np, results["objects"], results["texts"])
    st.image(annotated, caption="Annotated Result", use_column_width=True)

    # Show object summary
    st.subheader("🟢 Objects Detected")
    for obj in results["objects"]:
        st.markdown(f"- **{obj['label']}** (Conf: {obj['label_confidence']*100:.1f}%)")

    # Show text summary
    st.subheader("🔵 Text Detected")
    if results["texts"]:
        for t in results["texts"]:
            st.markdown(f"- \"{t['text']}\" (Conf: {t['confidence']*100:.1f}%)")
    else:
        st.markdown("No text detected.")

    # Download annotated image
    annotated_pil = Image.fromarray(annotated)
    img_bytes = annotated_pil.convert("RGB").tobytes("jpeg", "RGB")
    st.download_button("Download Annotated Image", data=annotated_pil.tobytes(), file_name="annotated.jpg", mime="image/jpeg")
