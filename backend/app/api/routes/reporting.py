from fastapi import APIRouter, HTTPException

from app.core.report import combine_book_titles_with_min_distance, generate_report_by_genre, generate_report_by_number
from app.models import ReportType

router = APIRouter()


@router.post('/generate-by-type')
def generate_report(report_type: ReportType, raw_detected_text_data, horizontal_epsilon=None, vertical_spacing_factor=None):
    grouped_result = combine_book_titles_with_min_distance(
        raw_detected_text_data,
        horizontal_epsilon,
        vertical_spacing_factor
    )
    match report_type:
        case ReportType.genre_distinction:
            return {'genre': generate_report_by_genre(grouped_result)}
        case ReportType.number_distinction:
            return {'number': generate_report_by_number(grouped_result)}
        case ReportType.all:
            genre_report = generate_report_by_genre(grouped_result)
            number_report = generate_report_by_number(grouped_result)
            return {'genre': genre_report, 'number': number_report}
        case _:
            raise HTTPException(
                status_code=400,
                detail='[Report] Incorrect type requested'
            )
