import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import pyarrow

class Batch_TextRead(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        path=params['path']
        text_dict=dict()
        if "hdfs://" not in path:
            if not os.path.exists(path):
                raise Exception("文件路径填写错误")
            if os.path.isdir(path):
                file_list=os.listdir(path)
                for file_name in file_list:
                    file_path = os.path.join(path,file_name)
                    if os.path.isfile(file_path):
                        with open(file_path,'r') as f:
                            text_dict[file_name]=f.read()
            else:
                file_name=path.split('/')[-1]
                with open(path, 'r') as f:
                    text_dict[file_name] = f.read()
        else:
            ip = path.split(':')[1][2:]
            tmp = path.split(':')[2]
            port = tmp[:tmp.index('/')]
            file_path = tmp[tmp.index('/'):]
            hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
            if not hdfs.exists(file_path):
                raise Exception("hdfs文件路径填写错误")
            # 把hdfs上面的文件夹先复制到本地
            folder_name = file_path.split('/')[-1]
            local_tmp_path = "/tmp/flok-tmp/" + folder_name
            os.system("hadoop fs -cp %s file://%s" % (path, local_tmp_path))
            # 在本地文件系统操作文件
            if os.path.isfile(local_tmp_path):
                with open(local_tmp_path,'r') as f:
                    text_dict[folder_name]=f.read()
            else:
                file_list = os.listdir(local_tmp_path)
                for file_name in file_list:
                    if '.crc' not in file_name:
                        full_name = os.path.join(local_tmp_path, file_name)
                        if os.path.isfile(full_name):
                            with open(full_name,'r') as f:
                                text_dict[file_name]=f.read()
            # 删除本地文件
            os.system("rm -r " + local_tmp_path)
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result

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
    algorithm = Batch_TextRead()
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)