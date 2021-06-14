import itertools
import os
import detectron2
from detectron2.structures import BoxMode
import cv2

# convert CCPD dataset to detectron2 format
def getDataset(d, dir):
    dataset_dicts = []
    url = dir + d
    for id, file in enumerate(os.listdir(url)):
        if d == 'train':
            if id == 2000:
                return dataset_dicts
        elif id == 1000:
            return dataset_dicts
        filename = file.split('.')[0]
        record = {}
        record['file_name'] = os.path.join(url,file)
        height, width = cv2.imread(record['file_name']).shape[:2]
        newlist = filename.split('-')[2].split('_')
        x1,y1 = newlist[0].split('&')
        x2,y2 = newlist[1].split('&')
        poly = [
            (int(x1), int(y1)), (int(x2), int(y1)),
            (int(x2), int(y2)), (int(x1), int(y2))
        ]
        poly = list(itertools.chain.from_iterable(poly))
        record['image_id'] = id
        record["height"] = height
        record["width"] = width
        obj = {
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": [poly],
        "category_id": 0,
        }
        record['annotations'] = [obj]
        dataset_dicts.append(record)

# convert Yolo dataset to detectron2 format
def getDataset(d, dir):
  dataset_dicts = []
  url = dir
  for index, file in enumerate(os.listdir(url)):
    if 'jpg' in file:
      continue
    elif 'txt' in file:
      name = file.split('.')[0] + '.jpg'
      height, width = cv2.imread(os.path.join(url,name)).shape[:2]
      filename = os.path.join(url,name)
      record = {}
      objs = []
      record['file_name'] = filename
      record['image_id'] = index
      record["height"] = height
      record["width"] = width
      with open(file,'r') as f:
        for line in f:
          id, xn, yn, wn, hn = [float(w) for w in line.split()] 
          widthAbs = wn * width
          heightAbs = hn * height
          x1 = xn * width - widthAbs/2
          y1 = yn * height - heightAbs/2
          x2 = x1 + widthAbs
          y2 = y1 + heightAbs
          poly = [
                (int(x1), int(y1)), (int(x2), int(y1)),
                (int(x2),int(y2)), (int(x1),int(y2))
          ]
          poly = list(itertools.chain.from_iterable(poly))
          obj = {
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": int(id)
          }
          objs.append(obj)
          record['annotations'] = objs
        dataset_dicts.append(record) 
    return dataset_dicts