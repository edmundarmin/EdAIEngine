import cv2,time

from edInferenceEngine.detection import yolov5 as engine
# from edInferenceEngine.detection import yolox as engine
from edInferenceEngine.ed_utils import image_resize,vis,COCO_CLASSES

classes = COCO_CLASSES

model = engine.getModel("edInferenceEngine\detection\yolov5\pretrained\yolov5s.pt",(640,640),classes=classes,device='cpu')

video = cv2.VideoCapture(0)

while 1:
    _,gambar = video.read()
    a = time.time()
    pred = model.inference(gambar)
    # print(1/(time.time()-a))
    if pred is not None:
        
        bboxes, scores, cls = pred
        gambar = vis(gambar, bboxes, scores, cls, 0.2, classes)

    cv2.imshow("asad",gambar)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
video.release()
cv2.destroyAllWindows()


