from src.utils import extract_zip_all
from src.CVAT2UltralyticsYOLO.cvat_coco import CVATCOCO2YOLO

ext = extract_zip_all(
    r'C:\Users\JiraponSasomsap\ArticulusProjects\tools\test_datasets', 
    r'C:\Users\JiraponSasomsap\ArticulusProjects\tools\unzip_test_datasets'
)

def id_mng(category_id):
    if category_id > 1:
        category_id = 2
    return category_id

ext = True
if ext:
    yolo = CVATCOCO2YOLO(
        dst=r'C:\Users\JiraponSasomsap\ArticulusProjects\tools\my-results',
        project='cvat_yolo',
        id_manager_func=id_mng
    )
    yolo.segment(
        src=r'C:\Users\JiraponSasomsap\ArticulusProjects\tools\unzip_test_datasets', 
        name='yolo-segmentation',
    ).auto_split(
        name='yolo-segmentation-split',
        train=0.8,
        val=0.2,
        test=0.0
    )