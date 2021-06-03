import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import numpy as np
import json


class Batch_ImageLuminanceConv(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        alpha = float(params.get('alpha'))
        gamma = float(params.get('gamma'))
        type = params.get('type')
        for image_name, image in image_dict.items():
            process_result = cv2.addWeighted(image, alpha, np.zeros(image.shape).astype(np.uint8), 1-alpha, gamma)
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
    #     "output": ["data/result5.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"alpha": "1","gamma":"50"}
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageLuminanceConv()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



