from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    center_x: float
    center_y: float
    height: float


def extract_coordinates(vertex_str: str) -> Tuple[float, float]:
    """Extract x, y coordinates from vertex string format '(x,y)'"""
    x, y = vertex_str.strip('()').split(',')
    return float(x), float(y)


def get_bounding_box(vertices: List[str]) -> BoundingBox:
    """Calculate bounding box with additional metrics from vertices"""
    coords = [extract_coordinates(v) for v in vertices]
    xs, ys = zip(*coords)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    height = max_y - min_y
    return BoundingBox(min_x, max_x, min_y, max_y, center_x, center_y, height)


def should_be_grouped(obj1: Dict, obj2: Dict,
                      horizontal_epsilon: float,
                      vertical_spacing_factor: float) -> bool:
    """
    Determine if two objects should be grouped based on both alignment and spacing

    Args:
        obj1: Dictionary objects containing vertices and descriptions
        obj2: Dictionary objects containing vertices and descriptions
        horizontal_epsilon: Maximum allowed horizontal distance between objects
        vertical_spacing_factor: Maximum allowed vertical spacing as a factor of text height
    """
    box1 = get_bounding_box(obj1['vertices'])
    box2 = get_bounding_box(obj2['vertices'])

    # Check horizontal alignment
    horizontal_aligned = abs(box1.center_x - box2.center_x) <= horizontal_epsilon

    # Check vertical spacing
    avg_height = (box1.height + box2.height) / 2
    max_vertical_spacing = avg_height * vertical_spacing_factor

    vertical_distance = min(
        abs(box1.max_y - box2.min_y),
        abs(box2.max_y - box1.min_y)
    )

    close_enough = vertical_distance <= max_vertical_spacing

    # Check if one object is numerical (like "83")
    is_number = lambda s: s.replace('.', '').isdigit()
    one_is_number = (is_number(obj1['description']) or
                     is_number(obj2['description']))

    # If one is a number, be more lenient with vertical spacing
    if one_is_number:
        max_vertical_spacing *= 1.5
        close_enough = vertical_distance <= max_vertical_spacing

    return horizontal_aligned and close_enough


def combine_book_titles_with_min_distance(
        raw_detected_text_data,
        horizontal_epsilon=50,
        vertical_spacing_factor=1.5) -> List[List[Dict]]:
    """
       Group related text objects based on spatial relationship and content type

       Args:
           raw_detected_text_data: List of dictionaries containing vertices and descriptions
           horizontal_epsilon: Maximum allowed horizontal distance between objects
           vertical_spacing_factor: Maximum allowed vertical spacing as a factor of text height

       Returns:
           List of groups, where each group contains related text objects
       """
    if not raw_detected_text_data:
        return []

    # Sort data by vertical position (top to bottom)
    sorted_data = sorted(raw_detected_text_data,
                         key=lambda x: get_bounding_box(x['vertices']).min_y)

    groups = []
    used = set()

    for i, obj1 in enumerate(sorted_data):
        if i in used:
            continue

        current_group = [obj1]
        used.add(i)

        # Look at subsequent objects for potential grouping
        for j in range(i + 1, len(sorted_data)):
            if j in used:
                continue

            obj2 = sorted_data[j]

            # Check if obj2 should be grouped with any object in current_group
            if any(should_be_grouped(existing_obj, obj2,
                                     horizontal_epsilon, vertical_spacing_factor)
                   for existing_obj in current_group):
                current_group.append(obj2)
                used.add(j)

        groups.append(current_group)

    # Sort objects within each group by vertical position
    for group in groups:
        group.sort(key=lambda x: get_bounding_box(x['vertices']).min_y)

    return groups


def generate_report_by_genre(grouped_result):
    return None


def generate_report_by_number(grouped_result):
    return None
