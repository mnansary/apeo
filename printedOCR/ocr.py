#-*- coding: utf-8 -*-
from __future__ import print_function
#-------------------------
# imports
#-------------------------
from .utils import localize_box,LOG_INFO
from .detector import Detector
from .skus import sku_names,sku_merged
from .masking import create_mask
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

    def process_boxes(self,img,line_boxes,crops):
        # line_boxes
        line_orgs=[]
        for bno in range(len(line_boxes)):
            tmp_box = copy.deepcopy(line_boxes[bno])
            x2,x1=int(max(tmp_box[:,0])),int(min(tmp_box[:,0]))
            y2,y1=int(max(tmp_box[:,1])),int(min(tmp_box[:,1]))
            line_orgs.append([x1,y1,x2,y2])
        
        # references
        line_refs=[]
        mask=create_mask(img,line_boxes)
        # Create rectangular structuring element and dilate
        mask=mask*255
        mask=mask.astype("uint8")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
        dilate = cv2.dilate(mask, kernel, iterations=4)

        # Find contours and draw rectangle
        cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            line_refs.append([x,y,x+w,y+h])
        line_refs = sorted(line_refs, key=lambda x: (x[1], x[0]))


        # organize       
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
        df,sorted_crops=self.process_boxes(img,line_boxes,crops)
        # recog
        res_eng = self.line_en.ocr(sorted_crops,det=False,cls=True)
        en_text=[i for i,_ in res_eng]
        df["text"]=en_text
        df=df[["line_no","word_no","text"]]

        lines=[]
        for line in tqdm(df.line_no.unique()):
            #_line=[]
            ldf=df.loc[df.line_no==line]
            ldf.reset_index(drop=True,inplace=True)
            ldf=ldf.sort_values('word_no')
            _ltext=''
            for idx in range(len(ldf)):
                text=ldf.iloc[idx,2]
                _ltext+='--'+text
            _ltext=_ltext.replace(" ",'')
            for ids,sku in enumerate(sku_merged):
                if sku in _ltext:
                    _ltext=_ltext.replace(sku,sku_names[ids])
                    lines.append(_ltext) 
            if "TOTAL" in _ltext:
                break

        prods=[]
        qtys=[]
        prices=[]

        for line in lines:
            line=line.split("-")
            line=[l for l in line if l.strip()]
            if len(line)==3:
                prods.append(line[0])
                qtys.append(line[1])
                prices.append(line[2])
                
        return df,pd.DataFrame({"sku":prods,"quantity":qtys,"price":prices})

        
        