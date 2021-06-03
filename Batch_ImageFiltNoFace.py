import sys, os
from FlokAlgorithmLocal import FlokDataFrame, FlokAlgorithmLocal
import cv2
import json


class Batch_ImageFiltNoFace(FlokAlgorithmLocal):
    def run(self, inputDataSets, params):
        image_dict = inputDataSets.get(0)
        faceClassifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        new_dict={}
        for image_name, image in image_dict.items():
            gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = faceClassifier.detectMultiScale(gray_img, 1.1, 3, cv2.CASCADE_SCALE_IMAGE)
            if len(faces)==0:
                continue
            new_dict[image_name] = image
        result = FlokDataFrame()
        result.addDF(new_dict)
        return result
if __name__ == "__main__":

    all_info = json.loads(sys.argv[1])
    params = all_info["parameters"]
    inputPaths = all_info["input"]
    inputTypes = all_info["inputFormat"]
    inputLocation = all_info["inputLocation"]
    outputPaths = all_info["output"]
    outputTypes = all_info["outputFormat"]
    outputLocation = all_info["outputLocation"]
    algorithm = Batch_ImageFiltNoFace()

    dataSet = algorithm.read(inputPaths,inputTypes,inputLocation,outputPaths,outputTypes)
    result = algorithm.run(dataSet, params)
    algorithm.write(outputPaths,result,outputTypes,outputLocation)



