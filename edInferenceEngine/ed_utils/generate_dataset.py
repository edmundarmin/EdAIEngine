import numpy as np







def imgbox2txt(shape,ic,bbox):

    h0,w0,c = shape

    text = ""
    ic = np.array(ic).astype(int)
    bbox = np.array(bbox).astype(float)

    print(ic,bbox)
    
    for data in zip(bbox,ic):
        x1,y1,x2,y2 = data[0]

        w = (x2-x1)/w0
        h = (y2-y1)/h0

        x = (x1/w0) + (w/2)
        y = (y1/h0) + (h/2)

        text+=f'{data[1]} {x} {y} {w} {h}\n'

    return text


    





