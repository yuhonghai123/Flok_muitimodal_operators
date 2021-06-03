import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import json


class Batch_ImageStrengthen(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        for image_name, image in image_dict.items():
            im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels_yuv = cv2.split(im_yuv)
            channels_yuv[0] = cv2.equalizeHist(channels_yuv[0])
            channels = cv2.merge(channels_yuv)
            process_result = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
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
    #     "output": ["data/result11.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {}#Sobel,Laplacian,Canny
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageStrengthen()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



