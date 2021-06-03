import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
from moviepy.editor import VideoFileClip


class Batch_VideoGetFrame(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        video_dict = inputDataSets.get(0)
        image_dict={}
        frame_time=float(params['time'])
        for video_name, video in video_dict.items():
            jpg = video.get_frame(frame_time)
            result_jpg = jpg[:, :, ::-1]
            image_name=video_name.split('.')[0]+"_frame.jpg"
            image_dict[image_name]=result_jpg
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["Whenever.mp4"],
    #         "inputFormat": ["video"],
    #         "inputLocation":["local_fs"],
    #         "output": ["test_getframe.jpg"],
    #         "outputFormat": ["pi"],
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
    algorithm = Batch_VideoGetFrame()

    videoList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(videoList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)