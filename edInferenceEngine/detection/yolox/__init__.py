from .models import get_model,preproc,postprocess
import torch

yoloxtype = {
    "yoloxS":[0.33,0.50]
}

class getModel:
    def __init__(self,pretrained,size,classes,device,modeltype="yoloxS"):
        
        self.device = device
        self.modeltype = modeltype
        self.yoloxtype = yoloxtype
     
        self.modeldepth,self.modelwidth = self.yoloxtype[self.modeltype]

        self.model = get_model(self.modeldepth,self.modelwidth,len(classes))
        self.model.eval()
        weight = torch.load(pretrained, map_location="cpu")      
        self.model.load_state_dict(weight["model"])

        self.size = size
        self.classes = classes

    def raw_inference(self,img):
        img, r = preproc(img, self.size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        

        if self.device == "cuda":
            self.model.to(self.device)
            img = img.cuda()
        
        pred = self.model(img)
        return pred


    def inference(self,img):
        img, r = preproc(img, self.size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "cuda":
            self.model.to(self.device)
            img = img.cuda()
        
        pred = self.model(img)
       
        out = postprocess(pred,len(self.classes))
     
        output = out[0]
        
        if output is not None:
            output = output.cpu()
            bboxes = output[:, 0:4]
            bboxes /= r
            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            return bboxes, scores, cls




