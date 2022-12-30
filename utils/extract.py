# Extract features from image
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import numpy as np
import cv2
import torch
import os

from torch import nn

from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference_single_image
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.config import get_cfg
import pickle


data_path = 'demo/data/genome/1600-400-20'
vg_classes = []
with open(os.path.join(data_path, 'objects_vocab.txt')) as f:
    for object in f.readlines():
        vg_classes.append(object.split(',')[0].lower().strip())

MetadataCatalog.get("vg").thing_classes = vg_classes

cfg = get_cfg()
cfg.merge_from_file('./config.yaml')
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL_WEIGHTS = "./faster_rcnn_from_caffe_attr_original.pkl"

predictor = DefaultPredictor(cfg)

im = cv2.imread("demo/data/images/input.jpg")
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

NUM_OBJECTS = 36

def doit(raw_image, raw_boxes):
        # Process Boxes
    raw_boxes = Boxes(torch.from_numpy(raw_boxes).cuda())
    
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        print("Original image size: ", (raw_height, raw_width))
        
        # Preprocessing
        image = predictor.aug.get_transform(raw_image).apply_image(raw_image)
        print("Transformed image size: ", image.shape[:2])
        
        # Scale the box
        new_height, new_width = image.shape[:2]
        scale_x = 1. * new_width / raw_width
        scale_y = 1. * new_height / raw_height
        #print(scale_x, scale_y)
        boxes = raw_boxes.clone()
        boxes.scale(scale_x=scale_x, scale_y=scale_y)

        # ----
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)
        
        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)
        
        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [boxes]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        print('Pooled features size:', feature_pooled.shape)
        
        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)

        print(pred_class_logits.shape)
        pred_class_prob = nn.functional.softmax(pred_class_logits, -1)
        pred_scores, pred_classes = pred_class_prob[..., :-1].max(-1)
        
        # Detectron2 Formatting (for visualization only)
        roi_features = feature_pooled
        instances = Instances(
            image_size=(raw_height, raw_width),
            pred_boxes=raw_boxes,
            scores=pred_scores,
            pred_classes=pred_classes
        )
        
        return instances, roi_features



given_boxes = np.array([
    [1.7189e+02, 1.6335e+02, 6.3919e+02, 4.1045e+02],
    [1.9629e+01, 0.0000e+00, 5.6497e+02, 1.5761e+02],
    [3.9222e+02, 0.0000e+00, 6.3402e+02, 2.7783e+02],
    [3.6025e+01, 0.0000e+00, 5.5431e+02, 2.8221e+02],
    [1.5994e+02, 1.5115e+00, 3.5376e+02, 3.1772e+02],
    [2.9326e+02, 1.4786e+02, 3.2540e+02, 1.8938e+02],
    [0.0000e+00, 3.6491e+02, 4.3185e+02, 4.7849e+02],
    [1.9907e+01, 4.2409e+02, 4.5854e+02, 4.7957e+02],
    [4.9555e+00, 8.1566e+01, 2.3710e+02, 4.5235e+02],
    [5.5625e+02, 2.7353e+02, 6.0216e+02, 3.7322e+02],
    [9.2086e+01, 2.8328e+02, 3.2548e+02, 4.4708e+02],
    [1.7761e+02, 3.6624e+02, 4.5720e+02, 4.6956e+02],
    [1.7253e+02, 3.7161e+02, 6.4000e+02, 4.7876e+02],
    [2.7954e+02, 2.0651e+02, 3.4036e+02, 3.1626e+02],
    [1.9732e+02, 3.8317e+01, 6.4000e+02, 3.2455e+02],
    [2.7082e+02, 1.1922e+00, 5.8803e+02, 3.0338e+02],
    [6.5850e+00, 1.8703e+02, 3.0191e+02, 4.7954e+02],
    [0.0000e+00, 0.0000e+00, 2.2748e+02, 2.3305e+02],
    [2.4732e-01, 3.3860e+02, 3.1494e+02, 4.7737e+02],
    [2.0554e+02, 1.9619e+00, 6.4000e+02, 2.7667e+02]]
)

instances, features = doit(im, given_boxes)

print("Classes", instances.pred_classes)
