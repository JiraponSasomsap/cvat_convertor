from src.utils import extract_zip_all
from src.CVAT2UltralyticsYOLO.cvat_coco import CVATCOCO2YOLO

ext = extract_zip_all('zip', 'unzip')
if ext:
    yolo = CVATCOCO2YOLO()
    yolo.bounding_boxes('unzip', 'results')
    yolo.auto_split()
