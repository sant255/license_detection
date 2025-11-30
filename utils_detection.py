import os
import cv2
import numpy as np
from ultralytics import YOLO

# Load model lazily (so imports are fast)
_model = None

def load_model(weights_path=None, device='cpu'):
    """Load YOLO model once. weights_path optional."""
    global _model
    if _model is None:
        if weights_path and os.path.exists(weights_path):
            _model = YOLO(weights_path)
        else:
            # Use yolov8n pretrained if custom weights not provided
            _model = YOLO("yolov8n.pt")
    return _model

def detect_plates_from_image(image_bgr, model=None, conf=0.25, classes=None):
    """
    image_bgr: OpenCV BGR image (numpy array)
    model: ultralytics YOLO model instance
    Returns list of dicts: { 'box': [x1,y1,x2,y2], 'conf': float, 'crop': np.ndarray }
    """
    if model is None:
        model = load_model()

    # convert BGR to RGB as ultralytics expects typically
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = model(img_rgb, conf=conf, classes=classes, verbose=False)  # classes can be set to license-plate class if trained
    dets = []
    # results is an iterable of Results objects
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return dets

    # boxes.xyxy: tensor of shape (n,4)
    for box in boxes:
        # each box can be a ultralytics.objects.boxes.Box object
        try:
            xyxy = box.xyxy.cpu().numpy().astype(int).tolist()[0] if hasattr(box.xyxy, "cpu") else box.xyxy.tolist()
        except Exception:
            try:
                xyxy = list(map(int, box.xyxy))
            except Exception:
                continue
        # coordinates
        x1, y1, x2, y2 = xyxy
        conf_score = float(box.conf[0]) if hasattr(box, "conf") else float(0.0)
        # clip coordinates
        h, w = image_bgr.shape[:2]
        x1 = max(0, min(x1, w-1))
        x2 = max(0, min(x2, w-1))
        y1 = max(0, min(y1, h-1))
        y2 = max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image_bgr[y1:y2, x1:x2].copy()
        dets.append({"box":[x1,y1,x2,y2], "conf":conf_score, "crop":crop})
    return dets

def annotate_image(image_bgr, dets):
    """Draw boxes on image and return annotated BGR"""
    out = image_bgr.copy()
    for d in dets:
        x1,y1,x2,y2 = d['box']
        conf = d.get('conf', 0)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"plate {conf:.2f}"
        cv2.putText(out, label, (x1, max(15,y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return out
