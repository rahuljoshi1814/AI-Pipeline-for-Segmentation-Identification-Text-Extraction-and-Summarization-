import cv2

def extract_text(image_np, reader, min_confidence=0.5):
    """
    Args:
        image_np: np.ndarray, RGB image.
        reader: EasyOCR reader object.
        min_confidence: minimum OCR confidence to keep.
    Returns:
        List of dicts with keys: bbox, text, confidence.
    """
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    results = reader.readtext(image_bgr)

    text_data = []
    for (bbox, text, conf) in results:
        if conf < min_confidence:
            continue
        text_data.append({
            "bbox": bbox,  # list of 4 corner points
            "text": text,
            "confidence": conf
        })
    
    return text_data
