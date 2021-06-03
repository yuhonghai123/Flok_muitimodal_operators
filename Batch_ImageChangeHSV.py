import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import cv2

#继承算法模板FlokAlgorithmLocal，因此会包含算法模板的一些方法，比如read和write。
class Batch_ImageChangeHSV(FlokAlgorithmLocal):
    # 重写run方法，其中inputDatasets为FlokDataFrame类型，params为参数字典。
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        for image_name, image in image_dict.items():
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h = hsv[:, :, 0]
            s = hsv[:, :, 1]
            v = hsv[:, :, 2]
            h_new = cv2.add(h, float(params['色相']))
            s_new = cv2.add(s, float(params['饱和度']))
            v_new = cv2.add(v, float(params['明度']))
            hsv_new = cv2.merge([h_new, s_new, v_new])
            process_result = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
            image_dict[image_name] = process_result
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result

#在flok流程中会执行这个main函数
if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # 获取参数
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    # 构造算子类
    algorithm = Batch_ImageChangeHSV()
    # 先读取上个算子的输出，返回值是FlokDataFrame。
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    # 运行算子的处理逻辑
    result = algorithm.run(dataSet, params)
    # 将结果写入到磁盘，会由下一个算子.read()读取
    algorithm.write(outputPaths,result,outputTypes,outputLocation)