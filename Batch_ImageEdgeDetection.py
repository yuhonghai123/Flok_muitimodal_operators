import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import numpy as np
import skimage
from skimage import util
import json


class Batch_ImageEdgeDetection(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        type = params.get('type')
        if type == 'Sobel':
            for image_name, image in image_dict.items():
                im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                x = cv2.Sobel(im_gray, -1, 1, 0)
                y = cv2.Sobel(im_gray, -1, 0, 1)
                Scale_absX = cv2.convertScaleAbs(x)
                Scale_absY = cv2.convertScaleAbs(y)
                process_result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
                image_dict[image_name] = process_result
        elif type == 'Laplacian':
            for image_name, image in image_dict.items():
                blur = cv2.GaussianBlur(image, (3, 3), 0)
                laplacian = cv2.Laplacian(blur, -1, ksize=3)
                process_result = cv2.convertScaleAbs(laplacian)
                image_dict[image_name] = process_result
        elif type == 'Canny':
            for image_name, image in image_dict.items():
                blur = cv2.GaussianBlur(image, (5, 5), 0)
                process_result = cv2.Canny(blur, 50, 150)
                image_dict[image_name] = process_result
        else:
            raise Exception("No such way of denoising.")
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result

if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])

    # all_info = {
    #     "input": ["data/test.jpg"],
    #     "inputFormat": ["jpg"],
    #     "inputLocation":["local_fs"],
    #     "output": ["data/result8.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"type": "Canny"}#Sobel,Laplacian,Canny
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageEdgeDetection()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



