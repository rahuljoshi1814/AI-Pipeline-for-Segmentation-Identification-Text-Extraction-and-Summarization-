# 🔍 Vision AI: Image Segmentation, CLIP Labeling & OCR

This project is an end-to-end **AI pipeline** for analyzing images. It combines advanced segmentation, object labeling, and text extraction into a single streamlined tool.

## 🚀 Live App

👉 [Try it here!](https://5tx3ukt4bkktcqnokvuc38.streamlit.app/)

---

## 💡 Features

✅ **Object Segmentation** using Mask R-CNN (pre-trained on COCO dataset)  
✅ **Object Labeling** using OpenAI CLIP model (zero-shot labeling)  
✅ **OCR (Text Extraction)** using EasyOCR  
✅ **Clean, intuitive UI** with Streamlit — just upload and analyze in one click  
✅ **Downloadable annotated image** with bounding boxes and text overlays

---

## ⚙️ How it Works

1️⃣ **Upload an image** (JPG, JPEG, PNG, up to 200MB).  
2️⃣ The app automatically:
- Segments objects from the image.
- Assigns labels to each object using CLIP.
- Extracts text using OCR.
3️⃣ **Annotated result image** is shown with overlays.
4️⃣ **Summaries** of detected objects and text are listed clearly.
5️⃣ Download your **annotated image** directly from the app.

---

## 🧑‍💻 Tech Stack

- **PyTorch** & **Torchvision** — Mask R-CNN for segmentation
- **OpenAI CLIP** — object label refinement
- **EasyOCR** — text extraction
- **OpenCV & PIL** — image processing
- **Streamlit** — interactive web application

---

## 📄 Requirements

streamlit
torch
torchvision
opencv-python
Pillow
numpy
easyocr
git+https://github.com/openai/CLIP.git


---

## 💬 Future Enhancements

- Emotion detection module (for faces)
- Document layout analysis
- Option to process multiple images in batch mode
- Flexible custom label lists

---


## 🤝 Contributing

Pull requests are welcome! Feel free to suggest new ideas or improvements.

---

## ⭐ Acknowledgments

- [PyTorch](https://pytorch.org/)
- [CLIP by OpenAI](https://github.com/openai/CLIP)
- [EasyOCR](https://github.com/JaidedAI/EasyOCR)
- [Streamlit](https://streamlit.io)

### 🚀 **[Try it live now!](https://5tx3ukt4bkktcqnokvuc38.streamlit.app/)**


