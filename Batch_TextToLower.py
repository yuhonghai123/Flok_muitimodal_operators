import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import json

class Batch_TextToLower(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        text_dict = inputDataSets.get(0)
        for text_name,text in text_dict.items():
            process_result=text.lower()
            text_dict[text_name]=process_result
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

    algorithm = Batch_TextToLower()
    dataSet = algorithm.read(inputPaths, inputTypes, inputLocation, outputPaths, outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths, result, outputTypes, outputLocation)


