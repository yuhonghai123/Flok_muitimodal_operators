# encoding=utf-8
import sys, os, re
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import jieba
import jieba.analyse
from gensim.models import word2vec


class Batch_TextWord2Vec(FlokAlgorithmLocal):
    def array2str(self,arr):
        return " ".join([str(x) for x in arr])
    def run(self, inputDataSets, params):
        text_dict =inputDataSets.get(0)
        punctuation = "[’。，！…!#$)“%&\(”*+,-.'/:;<=>?@[\\]^_`{|}~+]"
        corpus = []
        for text in text_dict.values():
            text = re.sub(punctuation, " ", text)
            if params['type']=='CHN':
                corpus.append(list(jieba.cut(text)))
            elif params['type']=="ENG":
                corpus.append(text.split())
            else:
                raise Exception("请输入正确的语言类型")
        model = word2vec.Word2Vec(corpus, hs=1, min_count=1, window=10,size=int(params["vector_size"]))
        ans=""
        for key in model.wv.vocab:
            word = "{word}:{value}".format(word=key, value=self.array2str(model.wv[key]))
            ans = ans + word + "\n"
        text_dict={"word2vec.txt":ans}
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
    #     "output": ["data/word2vec_result.txt"],
    #     "outputFormat": ["txt"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"min_count":2 , "size":10}
    # }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_TextWord2Vec()

    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)

    algorithm.write(outputPaths, result, outputTypes, outputLocation)


