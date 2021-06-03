#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import cv2
from moviepy.editor import*

###从画面img中寻找人脸
def find_faces(img):
    #级联分类器进行人脸识别
    eys_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
    #颜色转换函数
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #检测gray中人脸
    faces = eys_cascade.detectMultiScale(gray, 1.2, 5)
    return faces


###从视频中截取含有人物的片段
def find_durations(clip):
    duration_list = []  # 存储片段时间列表
    start_time = 0      # 记录片段开始时间, 以毫秒为单位
    end_time = 0        # 记录片段结束时间, 以毫秒为单位
    for i, img in enumerate(clip.iter_frames(fps=20)): #遍历视频 且切分每20FPS为一段 
        faces = find_faces(img) #找到每一段里面的人脸出现时间
            
        # 若发现人物出现且未记录开始时间（即上一段不是人像片段）,记为开始时间
        if len(faces) >= 1 and start_time == 0:
            start_time = i / 20
    
        # 若已记录开始时间且人物消失,记为结束时间
        if start_time > 0 and len(faces) == 0:
            end_time = i / 20
            # 将开始和结束时间添加到片段时间列表中并重置开始时间和结束时间
            duration_list.append([start_time, end_time])
            # 重置开始时间和结束时间
            start_time = end_time = 0
    # 打印片段时间列表并返回
    return duration_list

class Batch_VideoFindWithHuman(FlokAlgorithmLocal):

    ###run函数
    def run(self, inputDataSets,params):
        video_dict = inputDataSets.get(0)
        final_dict = {}
        for video_name, video in video_dict.items():
            #clip = VideoFileClip(filename)-----video是clip
            durations = find_durations(video)
            Composed_clips=[]
            for d in durations:
                start_t, end_t = d
                Composed_clips.append(video.subclip(start_t, end_t))
            final_dict[video_name] = concatenate_videoclips(Composed_clips)
        result = FlokDataFrame()
        result.addDF(final_dict)
        return result

if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])

    # all_info = {
    #          "input": [],
    #          "inputFormat": [],
    #          "inputLocation": [],
    #          "output": ["data/result.bmp"],
    #          "outputFormat": ["bmp"],
    #          "outputLocation": ["local_fs"],
    #          "parameters": {"path": "data/lena_bmp.bmp",}
    #      }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_VideoFindWithHuman()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)

