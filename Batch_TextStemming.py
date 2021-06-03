import sys, os, re
import json
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer


class Batch_TextStemming(FlokAlgorithmLocal):

    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        porter_stemmer = PorterStemmer()
        punctuation = "[’。，！…!#$)“%&\(”*+,-.'/:;<=>?@[\\]^_`{|}~+]"
        for text_name,text in text_dict.items():
            process_result = []
            text = re.sub(punctuation, "", text)
            tokens = word_tokenize(text)  # 分词
            for word in tokens:
                stem_word = porter_stemmer.stem(word)
                process_result.append(stem_word)
            text_dict[text_name] = " ".join(process_result)
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
    #     "output": ["data/stemming_result.txt"],
    #     "outputFormat": ["txt"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {""}
    # }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_TextStemming()

    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


