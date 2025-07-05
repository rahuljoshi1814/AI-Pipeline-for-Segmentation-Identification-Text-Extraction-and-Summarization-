import torch
import numpy as np
import cv2

def segment_image(image_np, model, score_threshold=0.5):
    """
    Args:
        image_np: np.ndarray, RGB image.
        model: Mask R-CNN model.
        score_threshold: minimum confidence to keep an object.
    Returns:
        List of dicts with keys: crop, bbox, score, mask.
    """
    # Convert to tensor and normalize
    image_tensor = torch.from_numpy(image_np / 255.).permute(2, 0, 1).float().unsqueeze(0)
    image_tensor = image_tensor.to(next(model.parameters()).device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)[0]

    results = []
    for i in range(len(outputs["scores"])):
        score = outputs["scores"][i].item()
        if score < score_threshold:
            continue

        bbox = outputs["boxes"][i].cpu().numpy().astype(int)
        mask = outputs["masks"][i, 0].cpu().numpy()
        mask_bin = mask > 0.5

        x1, y1, x2, y2 = bbox
        crop = image_np[y1:y2, x1:x2].copy()

        # Optional: apply mask on crop
        mask_cropped = mask_bin[y1:y2, x1:x2]
        crop_masked = crop.copy()
        crop_masked[~mask_cropped] = 0

        results.append({
            "crop": crop_masked,
            "bbox": bbox.tolist(),
            "score": score,
            "mask": mask_bin
        })

    return results
