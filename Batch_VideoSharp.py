import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
from moviepy.editor import VideoFileClip


class Batch_VideoSharp(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        video_dict = inputDataSets.get(0)
        width=float(params['width'])
        height=float(params['height'])
        for video_name, video in video_dict.items():
            video_dict[video_name] = video.resize((width, height))
        result = FlokDataFrame()
        result.addDF(video_dict)
        return result



if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["data/Whenever.mp4"],
    #         "inputFormat": ["mp4"],
    #         "inputLocation":["local_fs"],
    #         "output": ["data/whenever_1000.mp4"],
    #         "outputFormat": ["mp4"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {"duration":100, "width":"1000", "height":"500"}
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_VideoSharp()

    videoList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(videoList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)