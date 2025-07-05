import torch
from PIL import Image
import numpy as np
import clip

candidate_labels = [
    "person", "dog", "cat", "laptop", "bicycle", "car",
    "book", "bottle", "cup", "phone", "chair", "table",
    "plant", "keyboard", "monitor", "pen", "bag"
]

def identify_objects(objects, clip_model, preprocess, candidate_labels, device="cpu"):
    """
    Args:
        objects: list of dicts from segmentation module (must include 'crop').
        clip_model: loaded CLIP model.
        preprocess: CLIP image preprocessor.
        candidate_labels: list of text labels to compare.
        device: device string.
    Returns:
        List of dicts with added 'label' and 'label_confidence'.
    """
    # Prepare text tokens
    text_tokens = clip.tokenize(candidate_labels).to(device)

    # Encode text
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # Process each object
    for obj in objects:
        crop_img = obj["crop"]
        pil_img = Image.fromarray(crop_img).convert("RGB")
        image_input = preprocess(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_prob, top_label_idx = similarity[0].topk(1)

        obj["label"] = candidate_labels[top_label_idx.item()]
        obj["label_confidence"] = top_prob.item()

    return objects
