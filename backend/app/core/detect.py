from fastapi import HTTPException, File, UploadFile, Body
import numpy as np
import cv2
import pytesseract
from PIL import Image
from langdetect import detect_langs
import spacy
import re
from collections import Counter


def get_ocr_text(img):
    try:
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")
    return text


def ocr_score(image):
    text = get_ocr_text(image)
    # Count valid words (you might need to adjust this for non-English texts)
    word_count = sum(word.isalpha() for word in text.split())

    # Try to detect the language and get a confidence score
    try:
        lang = detect_langs(text)
        lang_score = lang[0].prob if lang else 0
    except Exception:
        lang_score = 0

    return word_count, lang_score, text


def rotate_image(image, angle):
    return np.rot90(image, k=angle // 90)


def find_correct_orientation_custom(image):
    orientations = [0, 90, 180, 270]  # 0: original, 90: right, 180: upside down, 270: left
    results = []

    for angle in orientations:
        rotated = rotate_image(image, angle)
        word_count, lang_score, text = ocr_score(rotated)
        results.append((angle, word_count, lang_score, text))

    # Sort by word count (ascending), then by language score (descending)
    results.sort(key=lambda x: (x[2], -x[1]), reverse=True)

    return results[:2]  # Return the top 2 results


def find_correct_orientation(image):
    osd = pytesseract.image_to_osd(image)
    rotation = int([line for line in osd.splitlines() if 'Rotate:' in line][0].split(': ')[1])
    return rotation


async def read_file(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_image = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    except IOError:
        raise HTTPException(status_code=400, detail="Unprocessable file.")


def read_file_local(file_path):
    # Open the image file
    img = Image.open(file_path)

    # Convert the image to a NumPy array (which YOLOv5 expects)
    img = np.array(img)

    # Return the image
    return img


def get_spine_regions(results):
    # Get bounding boxes for detected objects (book spines)
    spine_regions = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        x_min, y_min, x_max, y_max = map(int, box)
        spine_regions.append({
            "bounding_box": [x_min, y_min, x_max, y_max],
            "confidence": float(conf)
        })
    return spine_regions


def preprocess_img(img, bounding_box):
    x_min, y_min, x_max, y_max = bounding_box
    cropped_spine = img[y_min:y_max, x_min:x_max]

    gray = cv2.cvtColor(cropped_spine, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 0)  # Use smaller kernel size or skip
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morph open to remove noise and invert image
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Smaller kernel size
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Morphological closing to connect gaps
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Remove small dots (noise)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 10  # Adjust as necessary to remove small dots
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(closing, [contour], -1, (0, 0, 0), -1)  # Remove small dots

    # invert = 255 - opening
    invert = 255 - closing
    return invert


def clean_text(text):
    # Remove special characters and digits at the start of the string
    text = re.sub(r'^[^a-zA-Z]+', '', text)
    # Remove special characters and digits at the end of the string
    text = re.sub(r'[^a-zA-Z]+$', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_likely_title(text):
    words = text.split()
    if len(words) < 2:
        return False

    # Check if most words start with a capital letter
    capital_ratio = sum(1 for word in words if word[0].isupper()) / len(words)

    # Ignore common words that might be lowercase in titles
    common_lowercase = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and'}
    adjusted_capital_ratio = sum(1 for word in words if word[0].isupper() or word.lower() in common_lowercase) / len(
        words)

    return (len(text) > 3 and
            capital_ratio > 0.5 and
            adjusted_capital_ratio > 0.8 and
            not text.isupper())  # Avoid all uppercase text


def extract_book_titles_nlp(ocr_text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(ocr_text)

    potential_titles = []

    for sent in doc.sents:
        chunks = list(sent.noun_chunks) + [ent for ent in sent.ents if ent.label_ in ['PERSON', 'ORG', 'WORK_OF_ART']]

        for chunk in chunks:
            clean_chunk = clean_text(chunk.text)
            if is_likely_title(clean_chunk):
                potential_titles.append(clean_chunk)

    return list(set(potential_titles))


async def extract_book_details_from_img(file: UploadFile = File(...), bounding_box: list = Body(...)):
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    if len(bounding_box) != 4:
        raise HTTPException(status_code=400,
                            detail="Invalid bounding box format. Expected [x_min, y_min, x_max, y_max]")

    # Read the image file
    img = await read_file(file)

    inverted_img = preprocess_img(img, bounding_box)

    most_likely_angle_rotated_img = rotate_image(inverted_img, 90)
    rotation = find_correct_orientation(inverted_img)
    rotated = rotate_image(inverted_img, rotation)

    most_likely_angle_text = get_ocr_text(most_likely_angle_rotated_img)
    tesseract_angle_text = get_ocr_text(rotated)

    book_titles = extract_book_titles_nlp(most_likely_angle_text)
    title = ' '.join(book_titles)

    book_titles2 = extract_book_titles_nlp(tesseract_angle_text)
    title2 = ' '.join(book_titles2)

    titles = [title for title in [title, title2] if title.strip()]

    if not titles:
        return []

    if len(titles) == 1:
        return titles

    if len(titles) == 2 and titles[0] == titles[1]:
        return [titles[0]]

    return titles
