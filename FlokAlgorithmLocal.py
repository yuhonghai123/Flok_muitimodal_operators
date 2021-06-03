import numpy as np
import pandas as pd
import pyarrow
# import pandavro as pdx
import os
import cv2
import librosa
from moviepy.editor import *
import pickle

class FlokDataFrame:
    def __init__(self):
        self.__dfList = []
        self.__counter = 0
        self.__modelInputPath = []
        self.__modelOutputPath = []

    def getSize(self):
        return len(self.__dfList)

    def getData(self):
        if len(self.__dfList) == 0:
            print("DataFrame is empty!")
        return self.__dfList

    def get(self, index):
        if index >= self.getSize():
            print("FloKDataFrame out of index:")
            print(index)
        return self.__dfList[index]

    def addDF(self, newdf):
        self.__dfList.append(newdf)

    def addModelInputPath(self, path):
        self.__modelInputPath.append(path)

    def addModelOutputPath(self, path):
        self.__modelOutputPath.append(path)

    def getModelInputPath(self, index):
        if index >= len(self.__modelInputPath):
            print("FloKDataFrame's model input paths list out of index:" + index)
        return self.__modelInputPath[index]

    def getModelOutputPath(self, index):
        if index >= len(self.__modelOutputPath):
            print("FloKDataFrame's model output paths list out of index:" + index)
        return self.__modelOutputPath[index]

    def getModelSize(self):
        return len(self.__modelOutputPath)

    def next(self):
        if self.__counter < len(self.__dfList):
            data = self.__dfList[self.__counter]
            self.__counter += 1
            return data
        else:
            print("write data: list index out of range")

class FlokAlgorithmLocal:
    def pandasDfToNpArray(self, pandasDfList):
        npArrayList = []
        for i in range(0, len(pandasDfList)):
            npArray = pandasDfList[i].values
            npArrayList.append(npArray)
        return npArrayList
    def npArrayToPandasDf(self, npArrayList):
        pandasDfList = []
        for i in range(0, len(npArrayList)):
            pandasDf = pd.DataFrame(npArrayList[i])
            pandasDfList.append(pandasDf)
        return pandasDfList
    def read(self, inputPaths, inputTypes, inputLocation, outputPaths, outputTypes):
        globalInputType = ""
        if len(inputPaths) > 0 and inputPaths[0] == "":
            return
        if len(inputPaths) != len(inputLocation):
            print("len(inputPaths) != len(inputLocation)")
            return
        if len(inputPaths) != len(inputTypes):
            if len(inputTypes) >= 1:
                globalInputType = inputTypes[0]
            else:
                print("InputType length need to be equal with InputPath length, or just set one input type")
        flokDM = FlokDataFrame()
        inputLen = len(inputPaths)
        for i in range(0, inputLen):
            if inputLocation[i] == "local_fs":
                if globalInputType == "":
                    globalInputType = inputTypes[i]
                if globalInputType == "csv":
                    data = pd.read_csv(inputPaths[i], sep='|')
                elif globalInputType == "parquet":
                    data = pd.read_parquet(inputPaths[i], engine='pyarrow')
                elif globalInputType == "orc":
                    data = pd.read_orc(inputPaths[i], engine='pyarrow')
                elif globalInputType == "avro":
                    pass  # data = pdx.read_avro(inputPaths[i])
                elif globalInputType == "model":
                    flokDM.addModelInputPath(inputPaths[i])
                elif globalInputType == "txt" or globalInputType == "json":
                    with open(inputPaths[i], 'rb') as f:
                        data = pickle.load(f)
                elif globalInputType == "jpg" or globalInputType == "png" or globalInputType == "bmp":
                    #读取pickle文件，将python对象取出来
                    with open(inputPaths[i],'rb') as f:
                        data=pickle.load(f)
                elif globalInputType == "mp3" or globalInputType == "wav" or globalInputType == "ogg":
                    with open(inputPaths[i], 'rb') as f:
                        data = pickle.load(f)
                elif globalInputType == "mp4" or globalInputType == "avi":
                    dot_id=inputPaths[i].rfind('.')
                    dir_path=inputPaths[i][:dot_id]
                    file_list=os.listdir(dir_path)
                    data={}
                    for file_name in file_list:
                        file_path = os.path.join(dir_path, file_name)
                        try:#有可能有其他非视频文件
                            video=VideoFileClip(file_path)
                            data[file_name]=video
                        except:
                            pass
                else:
                    print("data type not existing" + globalInputType)
            elif inputLocation[i] == "hdfs":
                if globalInputType == "":
                    globalInputType = inputTypes[i]
                if globalInputType == "csv":
                    data = Hdfs(inputPaths[i]).readhdfs(deli='|')
                elif globalInputType == "txt" or globalInputType == "json":
                    # 设置本地临时文件名
                    file_name = inputPaths[i].split('/')[-1]
                    local_tmp_path = "/tmp/flok-tmp/" + file_name
                    # 从hdfs复制到本地
                    os.system("hadoop fs -cp %s file://%s" % (inputPaths[i], local_tmp_path))
                    # 在本地把pickle转换为数据
                    with open(local_tmp_path, 'rb') as f:
                        data = pickle.load(f)
                    # 删除本地临时文件
                    os.system('rm -r' + local_tmp_path)
                elif globalInputType == "jpg" or globalInputType == "png" or globalInputType == "bmp":
                    #设置本地临时文件名
                    file_name = inputPaths[i].split('/')[-1]
                    local_tmp_path = "/tmp/flok-tmp/" + file_name
                    #从hdfs复制到本地
                    os.system("hadoop fs -cp %s file://%s" % (inputPaths[i], local_tmp_path))
                    #在本地把pickle转换为数据
                    with open(local_tmp_path,'rb') as f:
                        data=pickle.load(f)
                    #删除本地临时文件
                    os.system('rm -r'+local_tmp_path)
                elif globalInputType == "mp3" or globalInputType == "wav" or globalInputType == "ogg":
                    # 设置本地临时文件名
                    file_name = inputPaths[i].split('/')[-1]
                    local_tmp_path = "/tmp/flok-tmp/" + file_name
                    # 从hdfs复制到本地
                    os.system("hadoop fs -cp %s file://%s" % (inputPaths[i], local_tmp_path))
                    # 在本地把pickle转换为数据
                    with open(local_tmp_path, 'rb') as f:
                        data = pickle.load(f)
                    # 删除本地临时文件
                    os.system('rm -r' + local_tmp_path)
                elif globalInputType == "mp4" or globalInputType == "avi":
                    # 获取hdfs文件夹路径
                    dot_id=inputPaths[i].rfind('.')
                    hdfs_forder_path=inputPaths[i][:dot_id]
                    local_tmp_path = "/tmp/flok-tmp/" + hdfs_forder_path.split('/')[-1]
                    # 从hdfs复制到本地
                    os.system("hadoop fs -cp %s file://%s" % (hdfs_forder_path, local_tmp_path))
                    # 在本地读取所有文件
                    file_list = os.listdir(local_tmp_path)
                    data = {}
                    for file_name in file_list:
                        file_path = os.path.join(local_tmp_path, file_name)
                        try:  # 有可能有其他非视频文件
                            video = VideoFileClip(file_path)
                            data[file_name] = video
                        except:
                            pass
                    # 删除本地临时文件
                    os.system('rm -r' + local_tmp_path)
            else:
                print("inputLocation not existing")
            flokDM.addDF(data)
        outputLen = len(outputPaths)
        for i in range(0, outputLen):
            if outputTypes[i] == "model":
                flokDM.addModelOutputPath(outputPaths[i])
        return flokDM
    def write(self, outputPaths, outputData, outputTypes, outputLocation):
        if outputData is None:
            print("Algorithm without output data")
            return
        if (len(outputPaths) != outputData.getSize() + outputData.getModelSize()) or (
                len(outputPaths) != len(outputTypes)):
            print("OutData's number " + str(
                outputData.getSize() + outputData.getModelSize()) + " is not equals to output's number " + str(
                len(outputPaths)))
            return
        outputLen = len(outputPaths)
        if (outputLen != len(outputLocation)):
            print("outputPathLen is not equal to outputLocationLen")
            return
        for i in range(0, outputLen):
            outData = outputData.next()
            if outputLocation[i] == "local_fs":
                if outputTypes[i] == "csv":
                    dir = outputPaths[i].rfind("/")
                    dirpath = outputPaths[i][:dir]
                    if (os.path.exists(dirpath) is not True):
                        os.mkdir(dirpath)
                    outData.to_csv(outputPaths[i], sep="|", header=True, index=False)
                elif outputTypes[i] == "parquet":
                    outData.to_parquet(outputPaths[i], engine='pyarrow')
                elif outputTypes[i] == "avro":
                    outData.to_avro(outputPaths[i], outData)
                    # outData.to_parquet(outputPaths[i], engine = 'fastparquet')
                elif outputTypes[i] == "model":
                    print("how to write model file")
                elif outputTypes[i] == "txt" or outputTypes[i] == "json":
                    dir = outputPaths[i].rfind("/")
                    dirpath = outputPaths[i][:dir]
                    if not os.path.exists(dirpath):
                        os.mkdir(dirpath)
                    with open(outputPaths[i],'wb') as f:
                        pickle.dump(outData,f)
                #输出的类型只会是特定的某一种，在新建配置->存储配置可以查看。数据库为data_format。
                elif outputTypes[i] == "jpg" or outputTypes[i] == "png" or outputTypes[i] == "bmp":
                    #检测路径是否存在
                    dir = outputPaths[i].rfind("/")
                    dirpath = outputPaths[i][:dir]
                    if not os.path.exists(dirpath):
                        os.mkdir(dirpath)
                    # cv2.imwrite(outputPaths[i].replace(".output", "." + outputTypes[i]), outData)
                    #outData为python对象，将其序列化，并存储
                    with open(outputPaths[i],'wb') as f:
                        pickle.dump(outData,f)
                elif outputTypes[i] == "wav" or outputTypes[i] == "mp3" or outputTypes[i] == "ogg":
                    # 检测路径是否存在
                    dir = outputPaths[i].rfind("/")
                    dirpath = outputPaths[i][:dir]
                    if not os.path.exists(dirpath):
                        os.mkdir(dirpath)
                    # cv2.imwrite(outputPaths[i].replace(".output", "." + outputTypes[i]), outData)
                    # outData为python对象，将其序列化，并存储
                    with open(outputPaths[i], 'wb') as f:
                        pickle.dump(outData, f)
                elif outputTypes[i] == "mp4" or outputTypes[i] == "avi":
                    # 检测路径是否存在
                    dot_id=outputPaths[i].rfind('.')
                    dirpath = outputPaths[i][:dot_id]
                    if not os.path.exists(dirpath):
                        #创建多级dir
                        os.makedirs(dirpath)
                    # 将mp4文件写入路径。
                    for video_name,video in outData.items():
                        new_name=os.path.join(dirpath,video_name)
                        video.write_videofile(new_name)
                else:
                    print("data type not existing " + outputTypes[i])
            elif outputLocation[i] == "hdfs":
                if outputTypes[i] == "csv":
                    Hdfs(outputPaths[i]).writehdfs(deli='|', data=outData)
                elif outputTypes[i] == "txt" or outputTypes[i] == "json":
                    ip = outputPaths[i].split(':')[1][2:]
                    tmp = outputPaths[i].split(':')[2]
                    port = tmp[:tmp.index('/')]
                    file_path = tmp[tmp.index('/'):]
                    hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
                    # 检查文件夹是否存在，如果不存在创建
                    idofslash = file_path.rfind('/')
                    dir = file_path[:idofslash]
                    file_name = file_path[idofslash + 1:]
                    if not hdfs.exists(dir):
                        hdfs.mkdir(dir)
                    local_tmp_path = '/tmp/flok-tmp/' + file_name
                    with open(local_tmp_path, 'wb') as f:
                        pickle.dump(outData, f)
                    # 上传文件到hdfs
                    cmd = "hadoop fs -cp file://%s %s" % (local_tmp_path, outputPaths[i])
                    os.system(cmd)
                    # 删除本地文件
                    cmd = "rm -r " + local_tmp_path
                    os.system(cmd)
                elif outputTypes[i] == "jpg" or outputTypes[i] == "png" or outputTypes[i] == "bmp":
                    ip = outputPaths[i].split(':')[1][2:]
                    tmp = outputPaths[i].split(':')[2]
                    port = tmp[:tmp.index('/')]
                    file_path = tmp[tmp.index('/'):]
                    hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
                    #检查文件夹是否存在，如果不存在创建
                    idofslash=file_path.rfind('/')
                    dir=file_path[:idofslash]
                    file_name=file_path[idofslash+1:]
                    if not hdfs.exists(dir):
                        hdfs.mkdir(dir)
                    local_tmp_path='/tmp/flok-tmp/'+file_name
                    with open(local_tmp_path,'wb') as f:
                        pickle.dump(outData,f)
                    #上传文件到hdfs
                    cmd = "hadoop fs -cp file://%s %s" % (local_tmp_path, outputPaths[i])
                    os.system(cmd)
                    # 删除本地文件
                    cmd = "rm -r " + local_tmp_path
                    os.system(cmd)
                elif outputTypes[i] == "mp3" or outputTypes[i] == "wav" or outputTypes[i] == "ogg":
                    ip = outputPaths[i].split(':')[1][2:]
                    tmp = outputPaths[i].split(':')[2]
                    port = tmp[:tmp.index('/')]
                    file_path = tmp[tmp.index('/'):]
                    hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
                    # 检查文件夹是否存在，如果不存在创建
                    idofslash = file_path.rfind('/')
                    dir = file_path[:idofslash]
                    file_name = file_path[idofslash + 1:]
                    if not hdfs.exists(dir):
                        hdfs.mkdir(dir)
                    local_tmp_path = '/tmp/flok-tmp/' + file_name
                    with open(local_tmp_path, 'wb') as f:
                        pickle.dump(outData, f)
                    # 上传文件到hdfs
                    cmd = "hadoop fs -cp file://%s %s" % (local_tmp_path, outputPaths[i])
                    os.system(cmd)
                    # 删除本地文件
                    cmd = "rm -r " + local_tmp_path
                    os.system(cmd)
                elif outputTypes[i] == "mp4" or outputTypes[i] == "avi":
                    ip = outputPaths[i].split(':')[1][2:]
                    tmp = outputPaths[i].split(':')[2]
                    port = tmp[:tmp.index('/')]
                    file_path = tmp[tmp.index('/'):]
                    hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))
                    # 检查文件夹是否存在，如果不存在创建
                    dot_id = file_path.rfind('.')
                    dir = file_path[:dot_id]
                    #被保存的文件夹名字,在airflow_services里面生成
                    folder_name=dir.split('/')[-1]
                    if not hdfs.exists(dir):
                        hdfs.mkdir(dir)
                    local_tmp_path = '/tmp/flok-tmp/' + folder_name
                    os.mkdir(local_tmp_path)
                    #先写到本地
                    for video_name,video in outData.items():
                        new_name=os.path.join(local_tmp_path,video_name)
                        video.write_videofile(new_name)
                    # 上传文件到hdfs
                    # 获取hdfs文件夹整体路径
                    dot_id = outputPaths[i].rfind('.')
                    os.system("hadoop fs -cp file://%s %s" % (local_tmp_path+'/*', outputPaths[i][:dot_id]))
                    # 删除本地文件
                    os.system("rm -r " + local_tmp_path)
            else:
                print("outputLocation not existing")

    def run(self, inputDataSets, params):
        print("waiting for override")


class Hdfs():
    def __init__(self, path):
        ip = path.split(':')[1][2:]
        tmp = path.split(':')[2]
        port = tmp[:tmp.index('/')]
        self.filename = tmp[tmp.index('/'):]
        self.hdfs = pyarrow.hdfs.connect(host=ip, port=int(port))

    def put(self, filename, path, chunk=2 ** 16, replication=0):
        """ Copy local file to path in HDFS """
        with self.hdfs.open(path, 'wb', replication=replication) as target:
            with open(filename, 'rb') as source:
                while True:
                    out = source.read(chunk)
                    if len(out) == 0:
                        break
                    target.write(out)

    def getmerge(self, path, filename):
        """ Concat all files in path (a directory) to local output file """
        files = self.hdfs.ls(path)
        idx = 0
        with open(filename, 'wb') as fout:
            for apath in files:
                with self.hdfs.open(apath, 'rb') as fin:
                    data = fin.read().splitlines(True)
                    if (idx == 0 and len(data)!=0):
                        fout.writelines(data[0:])
                        idx+=1
                    else:
                        fout.writelines(data[1:])

    def readhdfs(self, deli):
        self.getmerge(path=self.filename, filename='tmp.csv')  # 获取制定目录下的所有文件，复制合并到本地文件
        df = pd.read_csv("tmp.csv", deli)
        os.system("rm tmp.csv")
        return df

    def writehdfs(self, deli, data):
        data.to_csv("tmp_write.csv", sep=deli, header=True, index=False)
        self.put("tmp_write.csv", path=self.filename)  # 将本地的文件上传
        os.system("rm tmp_write.csv")