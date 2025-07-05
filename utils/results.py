def build_results(image_name, objects, text_data):
    """
    Args:
        image_name: str, filename or identifier.
        objects: list of dicts (with bbox, label, confidence, etc.).
        text_data: list of dicts (bbox, text, confidence).
    Returns:
        Unified result dictionary.
    """
    result = {
        "image_name": image_name,
        "objects": [],
        "texts": []
    }

    for obj in objects:
        result["objects"].append({
            "bbox": obj["bbox"],
            "label": obj["label"],
            "label_confidence": obj["label_confidence"],
            "score": obj["score"]
        })
    
    for t in text_data:
        result["texts"].append({
            "bbox": t["bbox"],
            "text": t["text"],
            "confidence": t["confidence"]
        })

    return result
