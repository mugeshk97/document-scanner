import cv2
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import easyocr

def imageproc(img):
    try:
        _, mask = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((2,2),np.uint8)
        dilation = cv2.dilate(mask,kernel,iterations = 3)
    except Exception as e:
        print(f'[ERROR in imgproc]: {e}')
    return dilation

def tocr(img):
    text = None
    img = imageproc(img)
    try:
        text = reader.readtext(img, allowlist = '0123456789-')
        if text != []:
            text = text[0][1]
    except Exception as e:
        print(f'[ERROR in tocr]: {e}')
    return text

def detect(filename):
    df = {}
    df['Filename'] = filename
    image = cv2.imread(filename)
    classIds, confs, bbox = net.detect(image, confThreshold=0.8, nmsThreshold=0.8)
    try:
        for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
            img = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            label = str(classes[classId])
            text = tocr(img)   
            df[label] = text

    except Exception as e:
        print(f'[ERROR in yolo]: {e}')
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",type=str, help="Path to input directory")
    parser.add_argument("--model",type=str, help="Path to model", default='yolov4_5000.weights')
    parser.add_argument("--config",type=str, help="Path to config", default='yolov4.cfg')
    args = parser.parse_args()
    folder = args.input_dir

    net = cv2.dnn_DetectionModel(args.model, args.config)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputMean((0,0,0))
    net.setInputSwapRB(True)
    print(f'[INFO] Model Loaded {args.model, args.config}')
    classes = ['Document No','Liber','Page']
    reader = easyocr.Reader(['en'], gpu= True)
    files = os.listdir(folder)
    files.sort()

    data = []
    for file_id in tqdm(range(len(files))):
        try:
            df = detect(f'{folder}/'+files[file_id])
            if df != {}:
                data.append(df)
            dataframe = pd.DataFrame(data)
            dataframe.to_csv(f'{folder}.csv', index= False)
        except Exception as e:
            print(f'Error: {e} Filename: {files[file_id]}')