import torch
import torchvision
import clip
import easyocr
import streamlit as st

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_maskrcnn():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model

@st.cache_resource
def load_clip_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

@st.cache_resource
def load_easyocr_reader():
    reader = easyocr.Reader(['en'])
    return reader
