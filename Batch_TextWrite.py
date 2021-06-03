import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import pyarrow
import time

class Batch_TextWrite(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        path=params['path']
        text_dict=inputDataSets.get(0)
        if "hdfs://" not in path:
            if not os.path.exists(path):
                raise Exception("输出的指向的文件夹不存在")
            for text_name,text in text_dict.items():
                # file_name=os.path.join(path,time.strftime("image_%Y-%m-%d_%H:%M:%S_")+str(i))
                new_name = time.strftime('%Y_%m_%d_%H_%M_%S_') + text_name
                file_name = os.path.join(path, new_name)
                with open(file_name,'w') as f:
                    f.write(text)
        else:
            ip = path.split(':')[1][2:]
            tmp = path.split(':')[2]
            port = tmp[:tmp.index('/')]
            file_path = tmp[tmp.index('/'):]
            hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
            if not hdfs.exists(file_path):
                raise Exception("输出指向的文件夹不存在")
            # 先把文件写入本地
            folder_name = file_path.split('/')[-1]
            local_tmp_path = "/tmp/flok-tmp/" + folder_name
            os.mkdir(local_tmp_path)
            for text_name, text in text_dict.items():
                new_name = time.strftime('%Y_%m_%d_%H_%M_%S_') + text_name
                file_name = os.path.join(local_tmp_path, new_name)
                with open(file_name,'w') as f:
                    f.write(text)
            # 把本地文件夹复制到hdfs上,后面要加/*，否则是把文件夹整个复制到path中，作为其子文件夹。
            os.system("hadoop fs -cp file://%s %s" % (local_tmp_path + '/*', path))
            # 删除本地文件
            os.system("rm -r " + local_tmp_path)

if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])

    # all_info = {
    #          "input": [],
    #          "inputFormat": [],
    #          "inputLocation": [],
    #          "output": ["data/result.bmp"],
    #          "outputFormat": ["bmp"],
    #          "outputLocation": ["local_fs"],
    #          "parameters": {"path": "data/lena_bmp.bmp",}
    #      }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_TextWrite()
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)