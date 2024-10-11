from fastapi import APIRouter, HTTPException, File, UploadFile, Body, Form
from app.models import SpineRegionsResponse
from app.core.detect import extract_book_details_from_img, get_spine_regions, read_file
import torch
from app.api.helpers.main import get_best_model_path
import json

router = APIRouter()

weight_path = get_best_model_path()

model = torch.hub.load('ultralytics/yolov5', 'custom', path=weight_path)


@router.post('/', response_model=SpineRegionsResponse)
async def detect_book_spines(file: UploadFile = File(...)):
    """
       Detect book spines and return bounding box coordinates.
    """
    # Read the image file
    img = await read_file(file)

    # Detect book spines using YOLOv5
    results = model(img)

    # Get bounding boxes for detected objects (book spines)
    spine_regions = get_spine_regions(results)

    if not spine_regions:
        raise HTTPException(status_code=400, detail="No book spines detected")

    # Return bounding boxes to the user
    return {"spine_regions": spine_regions}


@router.post('/book')
async def extract_book_title(file: UploadFile = File(...), bounding_box: str = Form(...)):
    """
       Detect single book bounding box text and returns the title.
    """
    bounding_box_list = json.loads(bounding_box)

    titles = await extract_book_details_from_img(file, bounding_box_list)
    return {"titles": titles}


@router.post('/books')
async def extract_books_titles(file: UploadFile = File(...), bounding_boxes: list[list[int]] = Body(...)):
    """
      Detect multiple books bounding box text and return their titles.
      - `bounding_boxes`: List of bounding boxes, each in the format [x_min, y_min, x_max, y_max]
      """
    # Ensure bounding_boxes is a list and has valid values
    if not bounding_boxes or not all(len(box) == 4 for box in bounding_boxes):
        raise HTTPException(status_code=400, detail="Bounding boxes must be a list of [x_min, y_min, x_max, y_max]")

    # Read the image once and pass it to the helper function for each bounding box
    detected_books = []

    for bounding_box in bounding_boxes:
        # Use the helper function to extract text from each bounding box
        title = await extract_book_details_from_img(file, bounding_box)
        detected_books.append({
            "bounding_box": bounding_box,
            "title": title
        })

    return {"books": detected_books}
