import sys, os
import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import json
import pyarrow
import time

class Batch_ImageWrite(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        #这个path会从前端传入，是一个已经存在的文件夹路径，是绝对路径，图片会写入到这个路径。
        path=params['path']
        image_dict = inputDataSets.get(0)
        if "hdfs://" not in path:
            if not os.path.exists(path):
                raise Exception("输出的指向的文件夹不存在")
            #把字典中的图片文件全部写到路径上，其中文件名是“当前时间+原来的文件名”
            for image_name,image in image_dict.items():
                # file_name=os.path.join(path,time.strftime("image_%Y-%m-%d_%H:%M:%S_")+str(i))
                new_name=time.strftime('%Y_%m_%d_%H_%M_%S_')+image_name
                file_name=os.path.join(path,new_name)
                cv2.imwrite(file_name,image)
        else:
            ip = path.split(':')[1][2:]
            tmp = path.split(':')[2]
            port = tmp[:tmp.index('/')]
            file_path = tmp[tmp.index('/'):]
            hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
            if not hdfs.exists(file_path):
                raise Exception("输出指向的文件夹不存在")
            #先把文件写入本地
            folder_name = file_path.split('/')[-1]
            local_tmp_path = "/tmp/flok-tmp/" + folder_name
            os.mkdir(local_tmp_path)
            for image_name,image in image_dict.items():
                new_name=time.strftime('%Y_%m_%d_%H_%M_%S_')+image_name
                file_name=os.path.join(local_tmp_path,new_name)
                cv2.imwrite(file_name,image)
            # 把本地文件夹复制到hdfs上,后面要加/*，否则是把文件夹整个复制到path中，作为其子文件夹。
            cmd = "hadoop fs -cp file://%s %s" % (local_tmp_path+'/*',path)
            os.system(cmd)
            # 删除本地文件
            cmd = "rm -r " + local_tmp_path
            os.system(cmd)
        result = FlokDataFrame()
        return result

if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])

    # all_info = {
    #     "input": ["/tmp/flok/abcd"],
    #     "inputFormat": ["jpg"],
    #     "inputLocation":["local_fs"],
    #     "output": [""],
    #     "outputFormat": [""],
    #     "outputLocation": [""],
    #     "parameters": {"path": "hdfs://172.16.244.5:9000/图片输出",}
    #     }
    params = all_info["parameters"]

    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]

    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]

    algorithm = Batch_ImageWrite()
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



