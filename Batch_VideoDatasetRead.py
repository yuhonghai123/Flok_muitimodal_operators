import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import requests
import Batch_VideoRead

class Batch_VideoDatasetRead(FlokAlgorithmLocal):
    def run(self,inputDataSets,params,globals):
        #获取Flok的ip和端口
        FLOK_IP=globals['server_ip']
        FLOK_PORT=globals['server_port']
        #向Flok请求数据库的信息
        url = 'http://%s:%s/dataset/get_dataset_path_info' % (FLOK_IP,FLOK_PORT)
        r = requests.get(url, params={'dataset_name': params['dataset_name']}, timeout=5)
        info_dict = json.loads(r.text)
        if info_dict['msg'] != 'success':
            raise Exception("数据集名填写错误")
        info_dict = info_dict['data']
        dataset = info_dict['dataset']
        if dataset['type']!='video':
            raise Exception("数据集类型不是文本类型")
        datasource = info_dict['datasource']
        # 利用数据库信息填写path参数
        hdfs_prefix=""
        if datasource['type'] == 'HDFS':
            hdfs_prefix="hdfs://%s:%s" % (datasource['ip'],datasource['port'])
        pre_path = datasource['database_name']
        #确保以/开头
        if not pre_path or pre_path[0] != '/':
            pre_path = '/' + pre_path
        rear_path = dataset['path']
        #确保后面的不以/开头
        if rear_path and rear_path[0] == '/':
            rear_path = rear_path[1:]
        complete_path = os.path.join(pre_path, rear_path)
        #加上前缀的路径（前缀可为空）
        complete_path=hdfs_prefix+complete_path
        # 调用Batch_VideoRead类中run的函数。
        # 返回DataFrame
        algorithm=Batch_VideoRead.Batch_VideoRead()
        return algorithm.run(inputDataSets,{'path':complete_path})
if __name__ == "__main__":
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

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    globals=all_info['globals']
    algorithm = Batch_VideoDatasetRead()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params,globals)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)
