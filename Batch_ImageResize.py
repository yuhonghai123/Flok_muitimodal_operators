import sys, os
import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import json


class Batch_ImageResize(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        for image_name,image in image_dict.items():
            resized_image=cv2.resize(image, (0, 0), fx=float(params.get("fx")), fy=float(params.get("fy")), interpolation=cv2.INTER_NEAREST)
            image_dict[image_name]=resized_image
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

    algorithm = Batch_ImageResize()
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



