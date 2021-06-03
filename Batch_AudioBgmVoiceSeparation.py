# 重采样
import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import numpy as np
import librosa
import soundfile as sf

class Batch_AudioBgmVoiceSeparation(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        audio_dict = inputDataSets.get(0)
        output_dict = dict()
        type=params['type']
        for audio_name, audio in audio_dict.items():
            y = audio[0]
            sr = audio[1]
            S_full, phase = librosa.magphase(librosa.stft(y))
            S_filter = librosa.decompose.nn_filter(S_full,
                                                   aggregate=np.median,
                                                   metric='cosine',
                                                   width=int(librosa.time_to_frames(2, sr=sr)))
            S_filter = np.minimum(S_full, S_filter)
            margin_i, margin_v = 2, 10
            power = 2
            mask_i = librosa.util.softmask(S_filter,
                                           margin_i * (S_full - S_filter),
                                           power=power)
            mask_v = librosa.util.softmask(S_full - S_filter,
                                           margin_v * S_filter,
                                           power=power)

            output_fname = audio_name.split(".")[0]
            if type == '人声' or type == '人声和背景音':
                S_foreground = mask_v * S_full
                output_voice = librosa.istft(S_foreground * phase)
                output_dict[output_fname+"_vocal.mp3"] = (output_voice, sr)
            if type == '背景音' or type == '人声和背景音':
                S_background = mask_i * S_full
                output_bgm = librosa.istft(S_background * phase)
                output_dict[output_fname+"_bgm.mp3"] = (output_bgm, sr)

        result = FlokDataFrame()
        result.addDF(output_dict)
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
    algorithm = Batch_AudioBgmVoiceSeparation()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)