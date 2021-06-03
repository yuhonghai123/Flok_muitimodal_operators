import sys, os
import pandas as pd
import numpy as np
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import json
import pyarrow

#继承算法模板FlokAlgorithmLocal，因此会包含算法模板的一些方法，比如read和write。
class Batch_ImageRead(FlokAlgorithmLocal):
    #重写run方法，其中inputDatasets为FlokDataFrame类型，params为参数字典。
    def run(self, inputDataSets, params):
        #image_dict是文件名和文件内容的字典,具体可以看后面，它会作为一个FlokDataFrame中的一个数据。
        image_dict=dict()
        #从字典中获取参数，参数名随意，自行设计即可，参数可以在Flok系统中设置，并由用户将参数从前端传入
        path=params['path']#path是一个保存所有图片文件的文件夹的绝对路径
        if "hdfs://" not in path:
            if not os.path.exists(path):
                raise Exception("文件路径填写错误")
            if os.path.isfile(path):
                im=cv2.imread(path)
                file_name=path.split('/')[-1]
                #文件名+内容，此处内容是cv2读取的图片的ndarray。
                image_dict[file_name]=im
            else:
                file_list = os.listdir(path)
                for file_name in file_list:
                    full_name = os.path.join(path, file_name)
                    if os.path.isfile(full_name):
                        im=cv2.imread(full_name)
                        image_dict[file_name]=im
        else:
            ip = path.split(':')[1][2:]
            tmp = path.split(':')[2]
            port = tmp[:tmp.index('/')]
            file_path = tmp[tmp.index('/'):]
            hdfs= pyarrow.hdfs.connect(host=ip, port=int(port))
            # with hdfs.open(path, 'rb') as fin:
            #     im = cv2.imdecode(np.frombuffer(fin.read(), np.uint8), cv2.IMREAD_COLOR)
            if not hdfs.exists(file_path):
                raise Exception("hdfs文件路径填写错误")
            # 把hdfs上面的文件夹先复制到本地
            folder_name=file_path.split('/')[-1]
            local_tmp_path = "/tmp/flok-tmp/"+folder_name
            cmd = "hadoop fs -cp %s file://%s" % (path,local_tmp_path)
            os.system(cmd)
            #在本地文件系统操作文件
            if os.path.isfile(local_tmp_path):
                im = cv2.imread(local_tmp_path)
                image_dict[folder_name] = im
            else:
                file_list = os.listdir(local_tmp_path)
                for file_name in file_list:
                    if '.crc' not in file_name:
                        full_name = os.path.join(local_tmp_path, file_name)
                        if os.path.isfile(full_name):
                            im = cv2.imread(full_name)
                            image_dict[file_name] = im
            #删除本地文件
            cmd = "rm -r " + local_tmp_path
            os.system(cmd)
        # 构造FlokDataFrame，将字典数据存入其中，并返回。
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result
#在flok流程中会执行这个main函数
if __name__ == "__main__":
    #这里会读取参数。因为执行该文件时时通过命令行传参的形式实现的。比如 python <xxx.py> <params>，这些是Flok里面自动生成的。
    all_info = json.loads(sys.argv[1])

    # all_info = {
    #          "input": [],
    #          "inputFormat": [],
    #          "inputLocation": [],
    #          "output": ["/tmp/flok/abcd"],
    #          "outputFormat": ["jpg"],
    #          "outputLocation": ["local_fs"],
    #          "parameters": {"path": "/tmp/flok_data/图片",}
    #      }
    #获取参数
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    #构造算子类
    algorithm = Batch_ImageRead()
    #先读取上个算子的输出，返回值是FlokDataFrame。
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    #运行算子的处理逻辑
    result = algorithm.run(dataSet, params)
    #将结果写入到磁盘，会由下一个算子.read()读取
    algorithm.write(outputPaths, result, outputTypes, outputLocation)



