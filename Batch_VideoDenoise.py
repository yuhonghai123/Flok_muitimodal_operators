import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
from moviepy.editor import *
import cv2



def denoise(clip):
    fps = clip.fps
    duration = clip.duration

    img_all = []
    for i in range(int(duration * fps)):
        img_all.append(clip.get_frame(i / fps))


    assert len(img_all) > 0

    new_img_all = []
    for i in range(len(img_all)):
        target = cv2.fastNlMeansDenoisingColored(img_all[i], 10, 10, 7, 21)
        new_img_all.append(target)

    new_video = ImageSequenceClip(new_img_all, fps=fps)
    return new_video

def img_func(image):
    return cv2.fastNlMeansDenoisingColored(image, 10, 10, 7, 21)


class Batch_VideoDenoise(FlokAlgorithmLocal):
    def run(self, inputDataSets,params):
        video_dict = inputDataSets.get(0)
        for video_name, video in video_dict.items():
            video_dict[video_name] = video.fl_image(img_func)
        result=FlokDataFrame()
        result.addDF(video_dict)
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
    algorithm = Batch_VideoDenoise()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)
