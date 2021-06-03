import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json
import jieba
import re
from gensim.summarization.summarizer import summarize

class Batch_TextSummary(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        ratio = float(params.get('ratio'))
        language_type=params['type']
        for text_name,text in text_dict.items():
            if language_type=='CHN':
                # 用单个空格替换额外空格
                text = re.sub(r'\s+', ' ', text)
                # 分句
                sentences = re.split('(。|！|\!|\.|？|\?)', text)
                # 分词的分句
                split_sentences = []
                # 对于每一句
                for sent in sentences:
                # 分词
                    seg_list = jieba.cut(sent)
                # 分词用空格分隔，合起来存入
                    s = " ".join(list(seg_list))
                # 将每一句存入lists
                    split_sentences.append(s)
                # 将每一句拼合起来
                tokens = "".join(split_sentences)
                # 替换符号成英语格式
                tokens = tokens.replace('。','。. ').replace('！','！. ').replace('!','!. ').replace('？','？. ').replace('?','?. ')
                # 调用函数进行摘要
                summary = summarize(tokens,ratio=ratio)
                # 将符号换回中文格式
                result = summary.replace(' ','').replace('.','')
                # 存入dict
                text_dict[text_name] = result
            elif language_type=='ENG':
                if text=="" or len(text.split("."))<=2:
                    text_dict[text_name]=text
                else:
                    text_dict[text_name]=summarize(text,ratio)
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

    algorithm = Batch_TextSummary()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


