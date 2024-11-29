from fastapi import HTTPException, File, UploadFile, Body
import numpy as np
import cv2
import pytesseract
from PIL import Image
from langdetect import detect_langs
import spacy
import re
from surya.ocr import run_ocr
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor

langs = ["en"]
det_processor, det_model = load_det_processor(), load_det_model()
rec_model, rec_processor = load_rec_model(), load_rec_processor()


def get_ocr_text(img):
    try:
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    except pytesseract.TesseractError as e:
        print(f"Tesseract error: {e}")
        return ""
    return text


def get_ocr_text_surya(img):
    try:
        predictions = run_ocr([img], [langs], det_model, det_processor, rec_model, rec_processor)
    except Exception as e:
        print(f"Surya error: {e}")
        return ""
    return predictions


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

    # gray = cv2.cvtColor(cropped_spine, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                cv2.THRESH_BINARY_INV, 11, 2)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    #
    # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1)
    #
    # contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # min_area = 5
    # for contour in contours:
    #     if cv2.contourArea(contour) < min_area:
    #         cv2.drawContours(closing, [contour], -1, (0, 0, 0), -1)
    #
    # invert = 255 - closing
    # median_filtered = cv2.medianBlur(invert, 3)
    # denoised = cv2.fastNlMeansDenoising(median_filtered, h=30, templateWindowSize=7, searchWindowSize=21)
    return cropped_spine


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
            potential_titles.append(clean_chunk)
            # if is_likely_title(clean_chunk):
            #     potential_titles.append(clean_chunk)

    return list(set(potential_titles))


async def extract_book_details_from_img(
        file: UploadFile = File(...),
        bounding_box: list = Body(...),
        img_to_read=None,
        idx=None
):
    pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    if len(bounding_box) != 4:
        raise HTTPException(status_code=400,
                            detail="Invalid bounding box format. Expected [x_min, y_min, x_max, y_max]")

    img = img_to_read if img_to_read is not None else await read_file(file)
    curr_input = ','.join(map(str, bounding_box))

    try:
        inverted_img = preprocess_img(img, bounding_box)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail={e, curr_input})

    most_likely_angle_rotated_img = rotate_image(inverted_img, 90)
    # cv2.imwrite(str(idx) + '. ' + curr_input + '.jpg', most_likely_angle_rotated_img)

    most_likely_angle_text = get_ocr_text(most_likely_angle_rotated_img)
    book_titles = extract_book_titles_nlp(most_likely_angle_text)
    title = ' '.join(book_titles)

    return title


async def extract_book_details_from_img_surya(
        file: UploadFile = File(...),
        bounding_box: list = Body(...),
        img_to_read=None,
):
    if len(bounding_box) != 4:
        raise HTTPException(status_code=400,
                            detail="Invalid bounding box format. Expected [x_min, y_min, x_max, y_max]")

    img = img_to_read if img_to_read is not None else await read_file(file)
    curr_input = ','.join(map(str, bounding_box))

    try:
        inverted_img = preprocess_img(img, bounding_box)
    except Exception as e:
        raise HTTPException(status_code=400,
                            detail={e, curr_input})

    most_likely_angle_rotated_img = rotate_image(inverted_img, 90)

    most_likely_angle_text = get_ocr_text_surya(img)
    print(most_likely_angle_text)
    title = ' '.join(most_likely_angle_text)

    return title
