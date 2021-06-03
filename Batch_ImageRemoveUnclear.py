import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import cv2

class Batch_ImageRemoveUnclear(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        result_dict = dict()
        blur = params.get('threshhold')
        for image_name, image in image_dict.items():
            imageVar = cv2.Laplacian(image, cv2.CV_64F).var()
            if(imageVar<float(blur)):
                continue
            else:
                result_dict[image_name]=image
        result = FlokDataFrame()
        result.addDF(result_dict)
        return result
#在flok流程中会执行这个main函数
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
    algorithm = Batch_ImageRemoveUnclear()
    # 先读取上个算子的输出，返回值是FlokDataFrame。
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    # 运行算子的处理逻辑
    result = algorithm.run(dataSet, params)
    # 将结果写入到磁盘，会由下一个算子.read()读取
    algorithm.write(outputPaths,result,outputTypes,outputLocation)