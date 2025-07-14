import json
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import List
import numpy as np
import yaml
from ..utils import auto_split, make_path

class CVATCOCO2YOLO:
    BOUNDING_BOXES_OUTNAME = 'yolo-bbox'
    SEGMENTATION_OUTNAME = 'yolo-segmentation'
    
    def __init__(self, id_manager_func=None):
        self._id = []
        self._cls = []
        self.categories=None
        self.dst_res = []
        self.id_manager_func = id_manager_func

    def _gen_id(self, category_id):

        if callable(self.id_manager_func):
            category_id = self.id_manager_func(category_id)
        
        cate_name = self.categories.loc[self.categories['id'] == category_id, 'name'].values[0]
        if category_id not in self._id:
            self._id.append(category_id)
            self._cls.append(cate_name)
        return self._id.index(category_id)

    def coco_reader(self, src, dst, outname, classes_select: List[int], labels_format:str="xywh"):
        if isinstance(src, str): src = Path(src)
        if isinstance(dst, str): dst = Path(dst)

        for folder in src.glob('*'):
            json_path = folder / 'annotations' / 'instances_default.json'
            with open(json_path, 'r') as j:
                json_data = json.load(j)

            imgs_info = pd.DataFrame(json_data['images'])
            annotations = pd.DataFrame(json_data['annotations'])
            self.categories = pd.DataFrame(json_data['categories'])

            save_at = dst / outname / folder.name
            save_at.mkdir(parents=True, exist_ok=True)

            for _, im_info in tqdm(imgs_info.iterrows(), total=len(imgs_info), desc=f'Processing {folder.name}'):
                ann_sel = annotations[annotations['image_id'] == im_info['id']]
                df = ann_sel[ann_sel['category_id'].isin(classes_select)] if classes_select else ann_sel

                text = ''
                if not df.empty:
                    selected = df[outname.split('-')[1]]
                    for index, data in enumerate(selected):
                        
                        # config class here
                        gen_cls_id = self._gen_id(df['category_id'].values[index])
                        
                        box = np.array(data).reshape(-1, 2)
                        width, height = im_info['width'], im_info['height']

                        if labels_format == "xywh":
                            box = np.hstack([
                                (box[0]+(box[1]/2)) / [width, height],
                                box[1] / [width, height]
                            ]) # minx, miny, w, h -> xywh
                        elif labels_format == "xy":
                            box = box / [width, height]
                            box = box.reshape(-1)
                        else:
                            raise ValueError(f"Unsupported labels_format: '{labels_format}'")
                        
                        text += f"{gen_cls_id} " + ' '.join(f'{v:.6f}' for v in box) + '\n'

                label_file = save_at / 'labels' / f"{im_info['file_name'][:-4]}.txt"
                label_file.parent.mkdir(parents=True, exist_ok=True)
                with open(label_file, 'w') as lb:
                    lb.write(text)

                imsrc = src / folder.name / 'images' / im_info['file_name']
                imdst = save_at / 'images' / im_info['file_name']
                imdst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(imsrc, imdst)

            yml = {
                'nc': len(self._id),
                'names': {self._gen_id(i): str(name) for i, name in zip(self._id, self._cls)}
            }
            with open(save_at / 'classes.yaml', 'w') as f:
                yaml.dump(yml, f, sort_keys=False)
            
            self.dst_res.append(save_at)

    def bounding_boxes(self, src, dst, name, classes_select:List[int]=None):
        save_as = make_path(dst, name)
        self.coco_reader(src, save_as, self.BOUNDING_BOXES_OUTNAME, classes_select, labels_format="xywh")
        print(f'Bounding Boxes Save : {save_as}')
        return self
    
    def segment(self, src, dst, name, classes_select:List[int]=None):
        save_as = make_path(dst, name)
        self.coco_reader(src, save_as, self.SEGMENTATION_OUTNAME, classes_select, labels_format="xy")
        print(f'Segment Save : {save_as}')
        return self
    
    def auto_split(self, dst:Path, name:str, train=0.7, val=0.2, test=0.1):
        save_as = make_path(dst, name)
        paths = []
        for src in tqdm(self.dst_res, desc="Splitting datasets"):
            _dst = save_as / src.stem
            auto_split(src, _dst, train, val, test)
            paths.append(Path(src.stem) / 'images')

        yml = {
            'path': save_as.as_posix(),
            'train':[Path(path / 'train').as_posix() for path in paths],
            'val':[Path(path / 'val').as_posix() for path in paths],
            'test':[Path(path / 'test').as_posix() for path in paths],
            'nc': len(self._id),
            'names': {self._gen_id(i): str(name) for i, name in zip(self._id, self._cls)}
        }
        with open(save_as / f'{name}.yaml', 'w') as f:
            yaml.dump(yml, f, sort_keys=False)

        filename = f'{int(train * 100)}-{int(val * 100)}-{int(test * 100)}.txt'
        with open(save_as / filename, 'w') as ff:
            ff.write(
                f"train : {int(train * 100)}\n"
                f"val : {int(val * 100)}\n"
                f"test : {int(test * 100)}\n"     
            )
        print(f'AutoSplit Save : {save_as}')