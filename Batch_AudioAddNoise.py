import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import numpy as np


class Batch_AudioAddNoise(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        audio_dict = inputDataSets.get(0)
        factor=float(params['factor'])
        for audio_name, audio in audio_dict.items():
            y = audio[0]
            sr = audio[1]
            random_values = np.random.rand(len(y))
            #转为同一类型'float32',否则会是64，体积增大一倍，且前端无法播放
            random_values = np.array(random_values,dtype=y.dtype)
            y = y + factor * random_values
            audio_dict[audio_name] = (y, sr)
        result = FlokDataFrame()
        result.addDF(audio_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["in.mp3"],
    #         "inputFormat": ["audio"],
    #         "inputLocation":["local_fs"],
    #         "output": ["test1.wav"],
    #         "outputFormat": ["audio"],
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
    algorithm = Batch_AudioAddNoise()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)