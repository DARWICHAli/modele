from torchvision import models
from torch import nn
from PIL import Image
import torchvision.transforms as T
import cv2 
import matplotlib.pyplot as plt

class EncoderCNN(nn.Module):
    """Encoder inputs images and returns feature boxes.
    """
    
    
    def __init__(self):
        super(EncoderCNN, self).__init__()
        self.model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True) 
        
        self.model.eval()
        
        self.coco_instances = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
                          'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 
                          'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 
                          'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 
                          'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
                          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 
                          'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
                          'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

                
    def get_prediction(self, img_path, threshold):
        """
        get_prediction
            parameters:
            - img_path - path of the input image
            - threshold - threshold value for prediction score
            method:
            - Image is obtained from the image path
            - the image is converted to image tensor using PyTorch's Transforms
            - image is passed through the model to get the predictions
            - class, box coordinates are obtained, but only prediction score > threshold
                are chosen.
        """
        img = Image.open(img_path)
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        pred_class = [self.coco_instances[i] for i in list(pred[0]['labels'].numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
        pred_score = list(pred[0]['scores'].detach().numpy())
        pred_t = [pred_score.index(x) for x in pred_score if x>threshold][-1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return  img, pred_boxes, pred_class

   