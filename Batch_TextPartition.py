# -*-coding:utf-8 -*-
import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import pandas as pd
import nltk
import re
import jieba.analyse


class Batch_TextPartition(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        punctuation = "[’。，！…!#$)“%&\(”*+,-.'/:;<=>?@[\\]^_`{|}~+]"
        if params['type'] == "CHN":
            for text_name,text in text_dict.items():
                text=re.sub(punctuation,"",text)
                seg_list = jieba.cut(text)
                process_result = " ".join(seg_list)
                text_dict[text_name]=process_result
        elif params['type'] == 'ENG':
            for text_name,text in text_dict.items():
                text = text.lower()
                #分成句子列表
                sentence_list = nltk.sent_tokenize(text)
                #每个句子进行分词，生成二维列表
                sentence_token_list=[nltk.word_tokenize(sentence) for sentence in sentence_list]
                #process_result里面每个元素是token的用“ ”的合并。
                process_result=[]
                for sentence_token in sentence_token_list:
                    process_result.append(" ".join(sentence_token))
                text_dict[text_name] = "\n".join(process_result)
        else:
            raise Exception("请正确填写类型")
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding='utf-8')
    # all_info = json.load(f)
    # all_info = {
    #     "input": ["data/chinese.txt"],
    #     "inputFormat":["txt"],
    #     "inputLocation": ["local_fs"],
    #     "output": ["data/chinese_result.txt"],
    #     "outputFormat": ["txt"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"language":"CHN", "dict":"武汉热干面,共抗疫情,北京炸酱面,常德米粉"}
    # }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]

    algorithm = Batch_TextPartition()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


