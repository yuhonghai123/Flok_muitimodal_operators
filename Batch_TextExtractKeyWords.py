import sys
import os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import pyarrow
import jieba.analyse

class Batch_TextExtractKeyWords(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        topk = params['topK']
        keywords_dict = dict()
        text_dict = inputDataSets.get(0)
        for text_name, text in text_dict.items():
            keyword = jieba.analyse.extract_tags(
                text, topK=topk, withWeight=False, allowPOS=())
            keyword_str = ' '.join(keyword)
            keywords_dict[text_name] = keyword_str
        result = FlokDataFrame()
        result.addDF(keywords_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_TextExtractKeyWords()
    audioList = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(audioList, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)
