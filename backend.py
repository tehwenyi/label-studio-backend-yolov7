from label_studio_ml import model
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys, DATA_UNDEFINED_NAME

import cv2
from yolov7.yolov7 import YOLOv7

model.LABEL_STUDIO_ML_BACKEND_V2_DEFAULT = True

class YOLOv7Model(LabelStudioMLBase):
  def __init__(self, **kwargs):
    kwargs['hostname'] = "http://app:8000"
    super(YOLOv7Model, self).__init__(**kwargs)

    self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
        self.parsed_label_config, 'RectangleLabels', 'Image'
    )

    self.labels_in_config = set(self.labels_in_config)

    self.model = YOLOv7(
      weights='weights/weights.pt',
      cfg='cfg/cfg.yaml',
      bgr=False,
      gpu_device=0,
      model_image_size=640,
      max_batch_size=64,
      half=True,
      same_size=True,
      conf_thresh=0.25,
      trace=False,
      cudnn_benchmark=False,
    )
  
  def _get_image_url(self, task):
    image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)
    return image_url

  def predict(self, tasks, **kwargs):
    print("starting predictions")
    results = []
    all_scores= []
    print("labels in config:", self.labels_in_config)
    task = tasks[0]
    image_url = self._get_image_url(task)
    image_path = self.get_local_path(image_url)
    img = cv2.imread(image_path)
    img_width, img_height = get_image_size(image_path)
    
    # if one ndarray given, this returns a list (boxes in one image) of tuple (box_infos, score, predicted_class)
    pred_classes = set()
    dets = self.model.detect_get_box_in(img, box_format='ltrb', classes=None, buffer_ratio=0.0)
    for det in dets:
      bb, score, cls = det
      l, t, r, b = bb
      pred_classes.add(cls)
      
      # LS expects the x, y, width, and height of image annotations
      # to be provided in percentages of overall image dimension
      x = int(l) / img_width * 100
      y = int(t) / img_height * 100
      w = (int(r) - int(l)) / img_width * 100
      h = (int(b) - int(t)) / img_height * 100

      results.append({
        "from_name": self.from_name,
        "to_name": self.to_name,
        "type": "rectanglelabels",
        "score": score,
        "original_width": img_width,
        "original_height": img_height,
        "image_rotation": 0,
        "value": {
          "rotation": 0,
          "x": x,
          "y": y,
          "width": w,
          "height": h,
          "rectanglelabels": [cls]
        }
      })
      all_scores.append(score)
    
    if not (pred_classes.issubset(self.labels_in_config)):
      print(f"======== WARNING: The following predicted classes are not in labels config [{pred_classes - self.labels_in_config}] ")

    print(results) # debug purposes

    avg_score = sum(all_scores) / max(len(all_scores), 1)
    
    return [{
      'result': results,
      'score': avg_score
    }]
