from fastapi import APIRouter, HTTPException, Body

from app.core.report import (combine_book_titles_with_min_distance, get_titles_from_groups,
                             generate_report_by_genre, generate_report_by_number, create_number_report_payload)
from app.models import ReportType, GenerateReportRequest

router = APIRouter()


@router.post('/generate-by-type-for-single')
def generate_report(request: GenerateReportRequest = Body(...)):
    grouped_result = combine_book_titles_with_min_distance(
        request.raw_detected_text_data,
        request.horizontal_epsilon,
        request.vertical_spacing_factor
    )
    titles_from_groups = get_titles_from_groups(grouped_result)
    match request.report_type:
        case "genre_distinction":
            return {'genre': generate_report_by_genre(titles_from_groups, request.category_count)}
        case "number_distinction":
            [outliers, inliers, dominant_number] = generate_report_by_number(titles_from_groups, request.category_count)
            number_report = create_number_report_payload(
                titles_from_groups,
                outliers,
                inliers,
                dominant_number,
                request.category_count
            )
            return {'number': number_report}
        case "all":
            genre_report = generate_report_by_genre(titles_from_groups, request.category_count)
            [outliers, inliers, dominant_number] = generate_report_by_number(titles_from_groups, request.category_count)
            number_report = create_number_report_payload(
                titles_from_groups,
                outliers,
                inliers,
                dominant_number,
                request.category_count
            )
            return {'genre': genre_report, 'number': number_report}
        case _:
            raise HTTPException(
                status_code=400,
                detail='[Report] Incorrect type requested'
            )


@router.post('/generate-by-type-for-bulk')
def generate_report(
        report_type: ReportType,
        bulk_raw_data_with_cat_count,
        horizontal_epsilon=None,
        vertical_spacing_factor=None
):
    result = []
    for shelf in bulk_raw_data_with_cat_count:
        raw_detected_text_data = shelf['raw_detected_text_data']
        category_count = shelf['raw_detected_text_data']
        result.append(generate_report(
            report_type,
            raw_detected_text_data,
            category_count,
            horizontal_epsilon,
            vertical_spacing_factor
        ))
    return {'reports': result}
