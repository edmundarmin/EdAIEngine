import os

from .models.common import DetectMultiBackend

from .models.utils.general import (check_img_size, non_max_suppression, scale_boxes)
from .models.utils.plots import Annotator, colors
from .models.utils.torch_utils import select_device
from .models.utils.augmentations import letterbox
import numpy as np
import torch

class getModel:
    def __init__(self,pretrained,size,classes=None,device='cpu'):
    
        self.device = torch.device(device)
       
     

        self.model = DetectMultiBackend(pretrained, device=self.device, dnn=False, fp16=False)
      
        self.stride, self.names = self.model.stride, self.model.names
        self.imgsz = check_img_size((640, 640), s=self.stride) 

        self.size = size
        if classes == None:
            self.classes=self.names
        else:
            self.classes = classes

    def raw_inference(self,im0):
        im,ratio, (dw, dh) = letterbox(im0, self.imgsz, stride=self.stride)  # padded resize
        imshape = im.shape
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
     

        pred = self.model(im)
      
        return pred,ratio,imshape

    def inference(self,im0,conf_thres=0.25,iou_thres=0.45,max_det=1000):
        pred,ratio,imgshape = self.raw_inference(im0)
        pred = non_max_suppression(pred, conf_thres,iou_thres, max_det=max_det)[0]
        if len(pred):
            if self.device == 'cuda':
                pred = pred.cpu()
            
            bbox = scale_boxes(imgshape[0:2], pred[:, :4], im0.shape).round()
            scores = pred[:,4]
            cls = pred[:,5]
    
            return bbox,scores,cls
        else:
            return None
        
     
    def inference2(self,im0):
        pred,ratio,imgshape = self.raw_inference(im0)
        pred = non_max_suppression(pred, 0.5, 0.5, max_det=1000)
        for i, det in enumerate(pred):  # per image
            im0 =  im0.copy()
           
            annotator = Annotator(im0, line_width=3, example=str(self.names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                # print(im0.shape)
                # print(im0.shape[:2])
                det[:, :4] = scale_boxes(imgshape[0:2], det[:, :4], im0.shape).round()
                # print(det[:, :4])
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    label =  f'{self.names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
        im0 = annotator.result()
        return im0

