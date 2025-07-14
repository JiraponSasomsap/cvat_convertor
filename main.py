from src.utils import extract_zip_all
from src.CVAT2UltralyticsYOLO.cvat_coco import CVATCOCO2YOLO
from pathlib import Path

ext = extract_zip_all('zip', 'unzip')

save_as = Path(r'F:\articulus\PROJECTS\teenoi\datasets\for_branch_train\BBK')

def id_mng(category_id):
    if category_id > 1:
        category_id = 2
    return category_id

ext = True
if ext:
    yolo = CVATCOCO2YOLO(id_manager_func=id_mng)
    yolo.segment(src=save_as, dst=save_as, name='BBK-yolo')
    yolo.auto_split(dst=save_as, name='BBK_v1x_seg')