import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
from moviepy.editor import VideoFileClip, concatenate_videoclips


class Batch_VideoConcatnate(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        video_dict = inputDataSets.get(0)
        clip_list=list(video_dict.values())
        finalclip = concatenate_videoclips(clip_list)
        new_dict={}
        new_dict["拼接视频.mp4"]=finalclip
        result = FlokDataFrame()
        result.addDF(new_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["whenever_2x.mp4","whenever_cut1.mp4"],
    #         "inputFormat": ["video","video"],
    #         "inputLocation":["local_fs","local_fs"],
    #         "output": ["whenever_2x.mp4"],
    #         "outputFormat": ["video"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {
    #                        }
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_VideoConcatnate()

    videoList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(videoList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)