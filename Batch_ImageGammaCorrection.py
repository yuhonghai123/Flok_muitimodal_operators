import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import cv2
from skimage import exposure

# 继承算法模板FlokAlgorithmLocal，因此会包含算法模板的一些方法，比如read和write。


class Batch_ImageGammaCorrection(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        #O=(I**gamma)*factor，其中I为0~1，最后会把O转换为0到255
        gamma = float(params.get('gamma'))
        factor=float(params.get('factor'))
        for image_name, image in image_dict.items():
            process_result = exposure.adjust_gamma(image, gamma,factor)
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
    #     "output": ["data/noise.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"type": "gaussian"}#gaussian, localvar, poisson, salt, pepper, s&p, speckle
    #     }
    # 获取参数
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageGammaCorrection()
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)