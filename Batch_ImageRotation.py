import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import numpy as np
import json


class Batch_ImageRotation(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        for image_name,image in image_dict.items():
            h = image.shape[0]
            w = image.shape[1]
            Angle = float(params.get('angle'))
            Scale = float(params.get('scale'))
            center = cv2.getRotationMatrix2D((w / 2, h / 2), Angle, Scale)
            process_result = cv2.warpAffine(image, center, (w, h))
            image_dict[image_name]=process_result
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result
if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageRotation()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



