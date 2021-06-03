import pandas as pd
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import sys, os
import librosa
import numpy as np

class Batch_AudioFindValid(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        audio_dict=inputDataSets.get(0)

        for audio_name,audio in audio_dict.items():
            y=audio[0]
            sr=audio[1]
            energy = self.calEnergy(y)
            zeroCrossingRate = self.calZeroCrossingRate(y)
            #N是一个二元组，返回（start_frame，end_frame）,其中start_frame指开始的第一个帧,而一个帧包含255个小帧。
            N = self.endPointDetect(y, energy, zeroCrossingRate)
            length=len(energy)
            startpoint=(len(y)*N[0])//length
            endpoint=(len(y)*N[-1])//length
            if endpoint>len(y):
                endpoint=len(y)
            audio_dict[audio_name] = (y[startpoint:endpoint],sr)
            
        result = FlokDataFrame()
        result.addDF(audio_dict)
        return result
    
    def sgn(self,data):
        if data >= 0 :
            return 1
        else :
            return 0

    # 计算每一帧的能量 256个采样点为一帧
    def calEnergy(self,wave_data) :
        energy = []
        sum = 0
        for i in range(len(wave_data)) :
            sum = sum + wave_data[i] * wave_data[i]
            if (i + 1) % 256 == 0 :
                energy.append(sum)
                sum = 0
            elif i == len(wave_data) - 1 :
                energy.append(sum)
        return energy

    #计算过零率  256个采样点为一帧
    def calZeroCrossingRate(self,wave_data) :
        zeroCrossingRate = []
        sum = 0
        for i in range(len(wave_data)) :
            if i % 256 == 0:
                continue
            sum = sum + np.abs(self.sgn(wave_data[i]) - self.sgn(wave_data[i - 1]))
            if (i + 1) % 256 == 0 :
                zeroCrossingRate.append(float(sum) / 255)
                sum = 0
            elif i == len(wave_data) - 1 :
                zeroCrossingRate.append(float(sum) / 255)
        return zeroCrossingRate

    # 利用短时能量，短时过零率，使用双门限法进行端点检测
    def endPointDetect(self,wave_data, energy, zeroCrossingRate) :
        energyAverage = sum(energy) / len(energy)
        ML = sum(energy[:5]) / 5                        
        MH = energyAverage / 4  #较高的能量阈值
        ML = (ML + MH) / 4  #较低的能量阈值
        Zs = float(sum(zeroCrossingRate[:5])) / 5  #过零率阈值
        A = []
        B = []
        C = []
        # 首先利用较大能量阈值 MH 进行初步检测
        flag = 0
        for i in range(len(energy)):
            if len(A) == 0 and flag == 0 and energy[i] > MH :
                A.append(i)
                flag = 1
            elif flag == 0 and energy[i] > MH and i - 21 > A[-1]:
                A.append(i)
                flag = 1
            elif flag == 0 and energy[i] > MH and i - 21 <= A[-1]:
                # A = A[:len(A) - 1]
                A.pop()
                flag = 1

            if flag == 1 and energy[i] < MH :
                A.append(i)
                flag = 0
        # print("较高能量阈值，计算后的浊音A:" + str(A))

        # 利用较小能量阈值 ML 进行第二步能量检测
        for j in range(len(A)) :
            i = A[j]
            if j % 2 == 1 :
                while i < len(energy) and energy[i] > ML :
                    i = i + 1
                B.append(i)
            else :
                while i > 0 and energy[i] > ML :
                    i = i - 1
                B.append(i)
        # print("较低能量阈值，增加一段语言B:" + str(B))

        # 利用过零率进行最后一步检测
        for j in range(len(B)) :
            i = B[j]
            if j % 2 == 1 :
                while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs :
                    i = i + 1
                C.append(i)
            else :
                while i > 0 and zeroCrossingRate[i] >= 3 * Zs :
                    i = i - 1
                C.append(i)
        # print("过零率阈值，最终语音分段C:" + str(C))
        return C

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
    algorithm = Batch_AudioFindValid()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)