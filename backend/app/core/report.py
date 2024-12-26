from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
from collections import Counter


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


def generate_report_by_genre(titles_from_groups, category_count):
    return None


def extract_number_prefix(title):
    # Use regex to find the leading number in the title
    match = re.match(r"^(\d+)", title)
    return match.group(1) if match else None


def generate_report_by_number(combined_title_data_passed, category_count):
    """
    Classifies items into inliers and outliers based on the top-k dominant number prefixes.

    Args:
        combined_title_data_passed (list of dict): List of items containing 'current_calculated_title'.
        category_count (int): Number of top dominant numbers to consider for inliers.

    Returns:
        list: A list containing [outliers, inliers_groups, dominant_numbers].
              - outliers: Items not in any of the top-k dominant groups.
              - inliers_groups: A dictionary mapping each dominant number to its group of inliers.
              - dominant_numbers: List of the top-k dominant numbers.
    """
    # Step 1: Count frequency of number prefixes
    number_counts = Counter()
    for item in combined_title_data_passed:
        prefix = extract_number_prefix(item['current_calculated_title'])
        if prefix:
            number_counts[prefix] += 1

    # Step 2: Get the top-k dominant numbers
    dominant_numbers = [num for num, _ in number_counts.most_common(category_count)]

    # Step 3: Separate inliers and outliers based on the top-k dominant numbers
    inliers_groups = {num: [] for num in dominant_numbers}
    outliers = []

    for item in combined_title_data_passed:
        prefix = extract_number_prefix(item['current_calculated_title'])
        if prefix in dominant_numbers:
            inliers_groups[prefix].append(item)
        else:
            outliers.append(item)

    return [outliers, inliers_groups, dominant_numbers]


def get_titles_from_groups(grouped_result: List[List[Dict]]):
    result = [{} for _ in range(len(grouped_result))]  # Initialize each element as an empty dictionary
    for i, group in enumerate(grouped_result):
        result[i]['group_number'] = i + 1
        current_calculated_title = []
        current_title_metadata = {}
        current_title_raw = {}
        # Initialize combined bounding box coordinates
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')
        for j, item in enumerate(group):
            box = get_bounding_box(item['vertices'])
            current_title_metadata[j] = {'box': box}  # Initialize as a dictionary with 'box' key
            current_title_raw[j] = item
            current_calculated_title.append(item['description'])

            # Update combined bounding box coordinates
            min_x = min(min_x, box.min_x)
            max_x = max(max_x, box.max_x)
            min_y = min(min_y, box.min_y)
            max_y = max(max_y, box.max_y)

        result[i]['current_calculated_title'] = " ".join(current_calculated_title)
        result[i]['current_title_metadata'] = current_title_metadata
        result[i]['current_title_raw'] = current_title_raw
        result[i]['bounding_box'] = [min_x, min_y, max_x, max_y]
    return result


def create_number_report_payload(data, outliers, inliers, dominant_number, category_count):
    total_inliers_count = sum(len(group) for group in inliers.values())
    return {
        'total_books': len(data),
        'outliers_count': len(outliers),
        'inliers_count': total_inliers_count,
        'dominant_number': dominant_number,
        'category_count': category_count,
        'outliers': outliers,
        'inliers': inliers,
        'misplacement_rate': len(outliers) / len(data) if data else 0
    }
