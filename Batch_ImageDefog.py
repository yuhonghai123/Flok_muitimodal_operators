import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import json

class Batch_ImageDefog(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        out_dict = dict()
        for image_name,image in image_dict.items():
            im_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels_yuv = cv2.split(im_yuv)
            channels_yuv[0] = clahe.apply(channels_yuv[0])
            channels = cv2.merge(channels_yuv)
            img_defog = cv2.cvtColor(channels, cv2.COLOR_YCrCb2BGR)
            # img_defog = clahe.apply(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
            out_dict[image_name] = img_defog
        result = FlokDataFrame()
        result.addDF(out_dict)
        return result
if __name__ == "__main__":
    all_info = json.loads(sys.argv[1])
    # all_info = {
    #     "input": [],
    #     "inputFormat": [],
    #     "inputLocation":[],
    #     "output": ["defog.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation": ["local_fs"],
    #     "parameters": {"path_in":"fog.jpg","path_out":"defog.jpg"}
    #     }
    params = all_info["parameters"]

    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]

    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]

    algorithm = Batch_ImageDefog()
    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



