import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import json
import cv2

class Batch_VideoMirror(FlokAlgorithmLocal):
    # 重写run方法，其中inputDatasets为FlokDataFrame类型，params为参数字典。
    def run(self, inputDataSets, params):
        #inputDataSets是一个FlokDataFrame，它里面已经包含了读取出来的数据，也就是video_dict，用get获取它。
        video_dict = inputDataSets.get(0)
        type=params['type']
        # 从字典中获取参数，参数名随意，自行设计即可，参数可以在Flok系统中设置，并由用户将参数从前端传入
        if type=='水平':
            for video_name, video in video_dict.items():
                process_result = video.fx(vfx.mirror_x)
                video_dict[video_name] = process_result
        elif type=='竖直':
            for video_name, video in video_dict.items():
                process_result = video.fx(vfx.mirror_y)
                video_dict[video_name] = process_result
        elif type=='水平竖直':
            for video_name, video in video_dict.items():
                process_result = video.fx(vfx.mirror_x).fx(vfx.mirror_y)
                video_dict[video_name] = process_result
        else:
            raise Exception('类型填写错误，请正确填写类型')
        #构造FlokDataFrame，将字典数据存入其中，并返回。
        result = FlokDataFrame()
        result.addDF(video_dict)
        return result


if __name__ == '__main__':
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
    algorithm = Batch_VideoMirror()

    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)