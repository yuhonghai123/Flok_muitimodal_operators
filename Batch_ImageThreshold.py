import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import numpy as np
import json


class Batch_ImageThreshold(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        threshold = 127
        maxVal = 230
        type = params.get('type')
        if type == 'binary':
            for image_name, image in image_dict.items():
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                returned_thresh_value, process_result = cv2.threshold(im_gray, threshold, maxVal, cv2.THRESH_BINARY)
                image_dict[image_name] = process_result
        elif type == 'binary_invert':
            for image_name, image in image_dict.items():
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                returned_thresh_value, process_result = cv2.threshold(im_gray, threshold, maxVal, cv2.THRESH_BINARY_INV)
                image_dict[image_name] = process_result
        elif type == 'trunc':
            for image_name, image in image_dict.items():
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                returned_thresh_value, process_result = cv2.threshold(im_gray, threshold, maxVal, cv2.THRESH_TRUNC)
                image_dict[image_name] = process_result
        elif type == 'tozero':
            for image_name, image in image_dict.items():
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                returned_thresh_value, process_result = cv2.threshold(im_gray, threshold, maxVal, cv2.THRESH_TOZERO)
                image_dict[image_name] = process_result
        elif type == 'tozero_invert':
            for image_name, image in image_dict.items():
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                returned_thresh_value, process_result = cv2.threshold(im_gray, threshold, maxVal, cv2.THRESH_TOZERO_INV)
                image_dict[image_name] = process_result
        else:
            raise Exception('类型填写错误，请正确填写类型')
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result

if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])

    # all_info = {
    #     "input": ["data/test.jpg"],
    #     "inputFormat": ["jpg"],
    #     "inputLocation":["local_fs"],
    #     "output": ["data/result13.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"type": "tozero_invert"}#binary,binary_invert,trunc,tozero,tozero_invert
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageThreshold()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



