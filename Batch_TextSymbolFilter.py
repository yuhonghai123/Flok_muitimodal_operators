import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json, re
import xml.etree.cElementTree as ET
import pandas as pd


class Batch_TextSymbolFilter(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        clean_tag = re.compile(r'<[^>]+>', re.S)
        clean_symbol = re.compile("[+\.\!\/_,$%^*(+\"\')]+|[+——()?【】“”！，；：;:。？、~@#￥%……&*（）]+")
        for text_name, text in text_dict.items():
            text = re.sub(clean_tag, '', text)
            text = re.sub(clean_symbol, ' ', text)
            text_dict[text_name] = text
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding='utf-8')
    # all_info = json.load(f)
    # all_info = {
    #     "input": ["data/english.txt"],
    #     "inputFormat":["txt"],
    #     "inputLocation": ["local_fs"],
    #     "output": ["data/english_filter.txt"],
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

    algorithm = Batch_TextSymbolFilter()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


