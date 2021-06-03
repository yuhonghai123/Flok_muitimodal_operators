import sys, os
from pandas.io.parsers import count_empty_vals
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import re
import pandas as pd
import jieba

class Batch_TextWordCount(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict=inputDataSets.get(0)
        token_list=[]
        punctuation = '[’!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~+]'
        if params['type']=='CHN':
            for text in text_dict.values():
                re.sub(punctuation," ",text)
                token_list.extend(list(jieba.cut(text)))
        else:
            for text in text_dict.values():
                re.sub(punctuation," ",text)
                text=text.lower()
                token_list.extend(text.split())
        count = {}
        for word in token_list:
            count[word]=count.get(word,0)+1
        items = list(count.items())
        items.sort(key= lambda x:x[1],reverse= True)
        ans=[]
        for word,count in items:
            ans.append('%s:%s' % (word,count))
        ans="\n".join(ans)
        text_dict={"word_count.txt": ans}
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result

if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding='utf-8')
    #all_info = json.load(f)
    # all_info = {
    #     "input": ["data/english.txt"],
    #     "inputFormat":["txt"],
    #     "inputLocation": ["local_fs"],
    #     "output": ["data/wordcount.csv"],
    #     "outputFormat": ["csv"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"number":5}
    # }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_TextWordCount()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation, outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)


