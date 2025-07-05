import cv2
import numpy as np

def draw_annotations(image_np, objects, texts):
    """
    Args:
        image_np: np.ndarray, RGB image.
        objects: list of dicts (bbox, label, label_confidence).
        texts: list of dicts (bbox, text, confidence).
    Returns:
        Annotated RGB image as np.ndarray.
    """
    annotated = image_np.copy()

    # Draw objects
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        label = obj["label"]
        score = obj["label_confidence"]

        # Draw rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label text
        text_str = f"{label} ({score*100:.1f}%)"
        cv2.putText(annotated, text_str, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Draw OCR text regions
    for t in texts:
        bbox_points = np.array(t["bbox"]).astype(int)
        cv2.polylines(annotated, [bbox_points], isClosed=True, color=(255, 0, 0), thickness=2)

        # Optionally put recognized text nearby
        text_str = t["text"]
        x, y = bbox_points[0]
        cv2.putText(annotated, text_str, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    return annotated
