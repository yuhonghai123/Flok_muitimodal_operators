# 重采样
import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import librosa

class Batch_AudioCut(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        audio_dict=inputDataSets.get(0)
        start = float(params["start"])
        end = float(params["end"])
        for audio_name,audio in audio_dict.items():
            y=audio[0]
            sr=audio[1]
            audio_dict[audio_name] = (y[int(start * sr):int(end * sr)],sr)
        result = FlokDataFrame()
        result.addDF(audio_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["test.mp3"],
    #         "inputFormat": ["audio"],
    #         "inputLocation":["local_fs"],
    #         "output": ["test_cut.wav"],
    #         "outputFormat": ["audio"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {"start":10 , #demo,没对时间做容错处理
    #                         "end":35
    #                        }
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_AudioCut()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)