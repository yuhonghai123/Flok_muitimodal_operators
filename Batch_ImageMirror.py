import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import cv2

#继承算法模板FlokAlgorithmLocal，因此会包含算法模板的一些方法，比如read和write。
class Batch_ImageMirror(FlokAlgorithmLocal):
    # 重写run方法，其中inputDatasets为FlokDataFrame类型，params为参数字典。
    def run(self, inputDataSets, params):
        #inputDataSets是一个FlokDataFrame，它里面已经包含了读取出来的数据，也就是image_dict，用get获取它。
        image_dict = inputDataSets.get(0)
        # 从字典中获取参数，参数名随意，自行设计即可，参数可以在Flok系统中设置，并由用户将参数从前端传入
        type = params.get('type')
        if type=='水平':
            #对字典中的每一张图片进行处理，然后再将处理结果保存在字典中。
            for image_name, image in image_dict.items():
                process_result = cv2.flip(image,1)
                image_dict[image_name] = process_result
        elif type=='垂直':
            for image_name, image in image_dict.items():
                process_result = cv2.flip(image,0)
                image_dict[image_name] = process_result
        elif type=='水平垂直':
            for image_name, image in image_dict.items():
                process_result = cv2.flip(image,-1)
                image_dict[image_name] = process_result
        else:
            raise Exception('类型填写错误，请正确填写类型')
        #构造FlokDataFrame，将字典数据存入其中，并返回。
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result
#在flok流程中会执行这个main函数
if __name__ == "__main__":
    # 这里会读取参数。因为执行该文件时时通过命令行传参的形式实现的。比如 python <xxx.py> <params>，这些是Flok里面自动生成的。
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
    # 构造算子类
    algorithm = Batch_ImageMirror()
    # 先读取上个算子的输出，返回值是FlokDataFrame。
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    # 运行算子的处理逻辑
    result = algorithm.run(dataSet, params)
    # 将结果写入到磁盘，会由下一个算子.read()读取
    algorithm.write(outputPaths,result,outputTypes,outputLocation)