import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import cv2


class Batch_ImageRemoveNoise(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        type = params.get('type')
        if type=='Gaussian':
            for image_name, image in image_dict.items():
                process_result = cv2.GaussianBlur(image, ksize=(3,3), sigmaX=0)
                image_dict[image_name] = process_result
        elif type=='Mean':
            for image_name, image in image_dict.items():
                process_result = cv2.blur(image, ksize=(3,3))
                image_dict[image_name] = process_result
        elif type=='Median':
            for image_name, image in image_dict.items():
                process_result = cv2.medianBlur(image, ksize=3)
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
    #     "output": ["data/noise.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"type": "gaussian"}#gaussian, localvar, poisson, salt, pepper, s&p, speckle
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageRemoveNoise()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)