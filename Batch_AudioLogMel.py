import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import numpy as np
import librosa
# import cv2
from pandas import Series, DataFrame


class Batch_AudioLogMel(FlokAlgorithmLocal):
    def array2str(self,arr):
        return " ".join([str(x) for x in arr])
    def run(self, inputDataSets, params):
        audio_dict = inputDataSets.get(0)
        text_dict={}
        for audio_name, audio in audio_dict.items():
            y = audio[0]
            sr = audio[1]
            melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=int(params.get("vector_size")))
            logmelspec = librosa.amplitude_to_db(melspec)  # 转换到对数刻度
            text_name=audio_name.split('.')[0]+"_LogMelSpec.txt"
            #将向量变为行
            logmelspec = logmelspec.T
            ans_list=[self.array2str(vec) for vec in logmelspec]
            text_dict[text_name]="\n".join(ans_list)
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding = 'utf-8')
    # all_info = json.loads(f)
    # all_info = {
    #         "input": ["in.mp3"],
    #         "inputFormat": ["mp3"],
    #         "inputLocation":["local_fs"],
    #         "output": ["test_mel.csv"],
    #         "outputFormat": ["csv"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {"n_mels": 128
    #                        }
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_AudioLogMel()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)