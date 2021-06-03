# 音高修正
import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import librosa
import matplotlib.pyplot as plt


class Batch_AudioPitch(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        audio_dict = inputDataSets.get(0)
        n_steps = float(params["n_steps"])
        for audio_name, audio in audio_dict.items():
            y = audio[0]
            sr = audio[1]
            audio_dict[audio_name] = librosa.effects.pitch_shift(y, sr, n_steps=n_steps), sr
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
    #         "output": ["test_pitch.wav"],
    #         "outputFormat": ["audio"],
    #         "outputLocation": ['local_fs'],
    #         "parameters": {"n_steps":6 , #demo,没对时间做容错处理
    #                         "bins_per_octave":12
    #                        }
    #     }

    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_AudioPitch()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)
