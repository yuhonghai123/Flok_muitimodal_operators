import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import numpy as np
import json


class Batch_ImageColorConv(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        origin_color = params.get('origin')
        converted_color = params.get('converted')
        if origin_color == 'BGR':
            if converted_color == 'gray':
                for image_name, image in image_dict.items():
                    process_result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    image_dict[image_name] = process_result
            elif converted_color == 'YCbCr':
                for image_name, image in image_dict.items():
                    process_result = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                    image_dict[image_name] = process_result
            elif converted_color == 'negative':
                for image_name, image in image_dict.items():
                    process_result = 255-image
                    image_dict[image_name] = process_result
                # for i in range(input_im.shape[0]):
                #     for j in range(input_im.shape[1]):
                #         result_img[i, j] = (255-input_im[i,j][0],255-input_im[i,j][1],255-input_im[i,j][2])
                #         result_img[:,:,0]=255-result_img[:,:,0]
            else:
                raise Exception("Not supported type.")
        elif origin_color == 'gray':
            if converted_color == 'negative':
                for image_name, image in image_dict.items():
                    process_result = 255-image
                    image_dict[image_name] = process_result
                # for i in range(input_im.shape[0]):
                #     for j in range(input_im.shape[1]):
                #         result_img[i, j] = 255 - input_im[i, j]
            else:
                raise Exception("Not supported type.")
        else:
            raise Exception("Not supported type.")
        result = FlokDataFrame()
        result.addDF(image_dict)
        return result

if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])

    # all_info = {
    #     "input": ["data/test.jpg"],
    #     "inputFormat": ["jpg"],
    #     "inputLocation":["local_fs"],
    #     "output": ["data/result10.jpg"],
    #     "outputFormat": ["jpg"],
    #     "outputLocation":["local_fs"],
    #     "parameters": {"Origin": "BGR", "Converted":"YCbCr"}#Origin:BGR--Converted:YCbCr/gray/negative, Origin:gray--Converted:negative
    #     }
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageColorConv()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



