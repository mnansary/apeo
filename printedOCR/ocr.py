#-*- coding: utf-8 -*-
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from .utils import localize_box,LOG_INFO
from .detector import Detector
from paddleocr import PaddleOCR
import cv2
import copy
from tqdm import tqdm
import pandas as pd 
tqdm.pandas()
from time import time
#-------------------------
# class
#------------------------
class PrintedOCR(object):
    def __init__(self):
        self.line_en=PaddleOCR(use_angle_cls=True, lang='en',rec_algorithm='SVTR_LCNet')
        self.det=Detector()
        LOG_INFO("Loaded Model")

    def process_boxes(self,line_boxes,crops):
        # line_boxes
        line_orgs=[]
        line_refs=[]
        for bno in range(len(line_boxes)):
            tmp_box = copy.deepcopy(line_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            line_orgs.append([x1,y1,x2,y2])
            line_refs.append([x1,y1,x2,y2]) 
        
        # merge
        for lidx,box in enumerate(line_refs):
            if box is not None:
                for nidx in range(lidx+1,len(line_refs)):
                    x1,y1,x2,y2=box    
                    x1n,y1n,x2n,y2n=line_orgs[nidx]
                    dist=min([abs(y2-y1),abs(y2n-y1n)])
                    if abs(y1-y1n)<dist and abs(y2-y2n)<dist:
                        x1,x2,y1,y2=min([x1,x1n]),max([x2,x2n]),min([y1,y1n]),max([y2,y2n])
                        box=[x1,y1,x2,y2]
                        line_refs[lidx]=None
                        line_refs[nidx]=box
                        
        line_refs=[lr for lr in line_refs if lr is not None]
               
        data=pd.DataFrame({"words":line_orgs,"word_ids":[i for i in range(len(line_orgs))]})
        # detect line-word
        data["lines"]=data.words.apply(lambda x:localize_box(x,line_refs))
        data["lines"]=data.lines.apply(lambda x:int(x))
        # register as crop
        text_dict=[]
        sorted_crops=[]
        for line in data.lines.unique():
            ldf=data.loc[data.lines==line]
            _boxes=ldf.words.tolist()
            _bids=ldf.word_ids.tolist()
            _,bids=zip(*sorted(zip(_boxes,_bids),key=lambda x: x[0][0]))
            for idx,bid in enumerate(bids):
                _dict={"line_no":line,"word_no":idx,"crop_id":bid,"box":line_boxes[bid]}
                sorted_crops.append(crops[bid])
                text_dict.append(_dict)
        df=pd.DataFrame(text_dict)
        return df,sorted_crops

    def __call__(self,img_path):
        # -----------------------start-----------------------
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # text detection
        line_boxes,crops=self.det.detect(img,self.line_en)
        df,sorted_crops=self.process_boxes(line_boxes,crops)
        # recog
        res_eng = self.line_en.ocr(sorted_crops,det=False,cls=True)
        en_text=[i for i,_ in res_eng]
        df["text"]=en_text
        # formatting
        df['x1']=df.box.progress_apply(lambda x:int(x[0][0]))
        df["y1"]=df.box.progress_apply(lambda x:int(x[0][1]))
        df=df[["line_no","word_no","text","x1","y1"]]

        data=[]
        for line in tqdm(df.line_no.unique()):
            _line=[]
            ldf=df.loc[df.line_no==line]
            ldf.reset_index(drop=True,inplace=True)
            ldf=ldf.sort_values('word_no')
            for idx in range(len(ldf)):
                text=ldf.iloc[idx,2]
                x1=ldf.iloc[idx,3]
                y1=ldf.iloc[idx,4]
                _line.append({"text":text,"x1":x1,"y1":y1})
            data.append(_line)
        data={"data":data}
        return data

        
        