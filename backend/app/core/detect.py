from fastapi import HTTPException, File, UploadFile, Body
import numpy as np
import cv2
import pytesseract
from PIL import Image


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


async def extract_book_details_from_img(file: UploadFile = File(...), bounding_box: list = Body(...)):
    pytesseract.pytesseract.tesseract_cmd = r'/usr/local/bin/pytesseract'
    if len(bounding_box) != 4:
        raise HTTPException(status_code=400,
                            detail="Invalid bounding box format. Expected [x_min, y_min, x_max, y_max]")

    # Read the image file
    img = await read_file(file)

    # Extract the region of the image corresponding to the selected book spine
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

    # cv2.imwrite('cropped-processed-v2.jpg', invert)

    # Convert the cropped image to a format that Tesseract can process
    pil_image = Image.fromarray(invert)

    # Perform OCR on the cropped image
    ocr_text = pytesseract.image_to_string(pil_image, lang='eng', config='--psm 6')

    if not ocr_text.strip():
        raise HTTPException(status_code=400, detail="No text detected in the selected region")

    return ocr_text.strip()
