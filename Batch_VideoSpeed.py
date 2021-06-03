import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
from moviepy.editor import VideoFileClip


class Batch_VideoSpeed(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        video_dict = inputDataSets.get(0)
        times = float(params.get("times"))
        for video_name, video in video_dict.items():
            new_video = video.fl_time(lambda t: times * t, apply_to=['mask', 'audio'])  # 1.1表示调整速度
            new_video = new_video.set_duration(video.duration / times)  # 1.1表示调整速度
            video_dict[video_name] = new_video
        result = FlokDataFrame()
        result.addDF(video_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["data/whenever.mp4"],
    #         "inputFormat": ["mp4"],
    #         "inputLocation":["local_fs"],
    #         "output": ["data/whenever_2x.mp4"],
    #         "outputFormat": ["mp4"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {"times": 2}
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_VideoSpeed()

    videoList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(videoList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)