import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
from moviepy.editor import VideoFileClip


class Batch_VideoCut(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        video_dict = inputDataSets.get(0)
        start = float(params["start"])
        end = float(params["end"])
        for video_name, video in video_dict.items():
            video_dict[video_name] = video.subclip(start,end)
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
    #         "output": ["data/Whenever_cut1.mp4"],
    #         "outputFormat": ["mp4"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {"start":50,
    #                         "end":60
    #                        }
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_VideoCut()

    videoList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(videoList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)