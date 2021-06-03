import sys, os, re, math
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
from nltk import sent_tokenize, word_tokenize, FreqDist
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import pandas as pd
import json


class CalculationTfidf():

    def stop_words(self, tokens):  # 支持自定义的停用词典，去除分词和停用词
        words = []
        punctuation = "[’。，！…!#$)“%&\(”*+,-.'/:;<=>?@[\\]^_`{|}~+]"
        stopwords = ["is", "the", "a", "an", "i", "me"]
        for i in tokens:
            i = re.sub(punctuation, " ", i)
            all_words = i.split()
            new_words = []
            for j in all_words:
                if j not in stopwords:
                    new_words.append(j)
            words.append(new_words)
        return words

    def count_word(self, word_list):  # 进行词频统计
        word_count={}
        for word in word_list:
            if word in word_count:
                word_count[word]+=1
            else:
                word_count[word]=1
        return word_count

    def tf(self, word, word_list, count):  # 计算TF(word代表被计算的单词，word_list是被计算单词所在文档分词后的字典)
        tf_value = word_list.get(word) / count
        return tf_value

    # 统计含有该单词的文章数
    def count_sentence(self, word, corpus):
        exist_count = 0
        # print(corpus)
        for x in corpus:
            if word in x:
                exist_count += 1
        return exist_count

    # 计算IDF
    def idf(self, word, corpus):
        article_count=len(corpus)
        idf_value = math.log((article_count) / (self.count_sentence(word, corpus) + 1))
        return idf_value

    # 计算TF-IDF
    def tf_idf(self, tf_value, idf_value):
        tf_idf = tf_value * idf_value
        return tf_idf


class Batch_TextTF_IDF(FlokAlgorithmLocal):

    def run(self, inputDataSets, params):  # 验证
        text_dict = inputDataSets.get(0)
        Cal_tfidf = CalculationTfidf()
        corpus=[word_tokenize(text) for text in text_dict.values()]
        punctuation = "[’。，！…!#$)“%&\(”*+,-.'/:;<=>?@[\\]^_`{|}~+]"
        for text_name,text in text_dict.items():
            text = re.sub(punctuation, " ", text)
            tokens = word_tokenize(text)
            total_num = len(tokens)
            wordCount = Cal_tfidf.count_word(tokens)
            cur_sentance_ans=[]
            for word in tokens:
                word_num=wordCount[word]
                tf=word_num/total_num
                idf=Cal_tfidf.idf(word,corpus)
                cur_sentance_ans.append('%s:%s' % (word,tf*idf))
            text_dict[text_name]="\n".join(cur_sentance_ans)
        result = FlokDataFrame()
        result.addDF(text_dict)
        return result


if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # f = open("test.json", encoding='utf-8')
    # all_info = json.load(f)
    # all_info = {
    #     "input": ["data/english_tf.txt","data/english1.txt","data/english2.txt","data/english3.txt"], #第一个 要处理的文章,之后的文件为语料库文件
    #     "inputFormat":["txt","txt","txt","txt"],
    #     "inputLocation": ["local_fs","local_fs","local_fs","local_fs"],
    #     "output": ["data/tf_result.csv"],
    #     "outputFormat": ["csv"],
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
    algorithm = Batch_TextTF_IDF()

    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


