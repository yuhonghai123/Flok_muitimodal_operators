import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import numpy as np
import skimage
from skimage import util
import json


class Batch_ImageCut(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        #top大家一般输入高的值，bottom一般输入低的值。比如20%~90%。
        top_percent = 1-float(params.get('top'))
        bottom_percent =1-float(params.get('bottom'))
        left_percent = float(params.get('left'))
        right_percent= float(params.get('right'))
        for image_name, image in image_dict.items():
            h = image.shape[0]
            w = image.shape[1]
            current_top = int(h*top_percent)
            current_bottom =int(h*bottom_percent)
            current_left=int(w*left_percent)
            current_right=int(w*right_percent)
            process_result = image[current_top:current_bottom,current_left:current_right]
            image_dict[image_name] = process_result
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result

if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])

    # all_info = {
    #     "input": ["data/test.jpg"],
    #     "inputFormat": ["jpg"],
    #     "inputLocation":["local_fs"],
    #     "output": ["data/result9.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"top": "100","bottom":"300","left":"100","right":"300"}
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageCut()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



