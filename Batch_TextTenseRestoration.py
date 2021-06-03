import sys, os, re
from nltk.tag import pos_tag

from nltk.tokenize import word_tokenize
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


class Batch_TextTenseRestoration(FlokAlgorithmLocal):
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        wnl = WordNetLemmatizer()
        punctuation = "[’。，！…!#$)“%&\(”*+,-.'/:;<=>?@[\\]^_`{|}~+]"
        for text_name, text in text_dict.items():
            process_result = []
            text = re.sub(punctuation, "", text)
            tokens = word_tokenize(text)  # 分词
            tagged_sent = pos_tag(tokens)  # 获取单词词性
            for tag in tagged_sent:
                wordnet_pos = Batch_TextTenseRestoration.get_wordnet_pos(tag[1]) or wordnet.NOUN
                process_result.append(wnl.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
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
    #     "output": ["data/tense_result.txt"],
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
    algorithm = Batch_TextTenseRestoration()

    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


