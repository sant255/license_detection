import re
import easyocr
import numpy as np

# Create reader lazily
_reader = None

def get_reader(lang_list=['en']):
    global _reader
    if _reader is None:
        # GPU accél available if installed with GPU support
        _reader = easyocr.Reader(lang_list, gpu=False)
    return _reader

def read_plate_text_from_image(crop_image_bgr, langs=['en']):
    """
    crop_image_bgr: BGR numpy image (cropped plate)
    returns cleaned text string or empty string
    """
    reader = get_reader(langs)
    # convert to RGB
    if len(crop_image_bgr.shape) == 3:
        img_rgb = crop_image_bgr[:,:,::-1]
    else:
        img_rgb = crop_image_bgr
    # EasyOCR accepts numpy arrays
    results = reader.readtext(img_rgb)
    # join detected text pieces
    text = "".join([t[1] for t in results]).strip()
    # Clean common noise
    cleaned = clean_plate_text(text)
    return cleaned

def clean_plate_text(text):
    """
    Clean OCR text: remove non-alphanumerics, uppercase, fix O vs 0 confusion heuristically
    Add regex validation if you know the license format.
    """
    if not text:
        return ""
    # keep alphanumerics and hyphen
    s = re.sub(r'[^A-Za-z0-9\- ]+', '', text)
    s = s.upper().replace(' ', '')
    # Fix common confusions: O -> 0 only if surrounded by digits heuristically
    s = re.sub(r'(?<=\d)O(?=\d)', '0', s)
    s = re.sub(r'(?<=\d)I(?=\d)', '1', s)
    # Additional rules can be applied per country format
    return s