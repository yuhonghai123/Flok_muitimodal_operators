import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
from zhconv import convert
import json

class Batch_TextTransfer(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        type = params['type']
        type2param={
            '大陆简体':'zh-cn',
            '台湾正体':'zh-tw',
            '香港繁体':'zh-hk',
            '简体':'zh-hans',
            '繁体':'zh-hant'
        }
        out_dict = dict()
        for text_name,text in text_dict.items():
            #繁简体转换
            text_transfer=convert(text,type2param[type])
            print(text_transfer)
            #保存至out_dict
            out_dict[text_name]=text_transfer
        result = FlokDataFrame()
        result.addDF(out_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding='utf-8')
    # all_info = json.load(f)
    # all_info = {
    #     "input": ["data/english.txt"],
    #     "inputFormat":["txt"],
    #     "inputLocation": ["local_fs"],
    #     "output": ["data/english_lower.txt"],
    #     "outputFormat": ["txt"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {}
    # }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]

    algorithm = Batch_TextTransfer()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


