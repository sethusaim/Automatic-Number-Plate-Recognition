import base64
import json
import os
from wsgiref import simple_server

from flask import Flask, Response, request
from flask_cors import CORS

from getNumberPlateVals import detect_license_plate
from predict_images import DetectVehicleNumberPlate

app = Flask(__name__)

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")
CORS(app)

inputFileName = "inputImage.jpg"

imagePath = os.path.join("images", inputFileName)

image_display = True

pred_stagesArgVal = 2

croppedImagepath = os.path.join("images", "croppedImage.jpg")


class ClientApp:
    def __init__(self):
        self.modelArg = os.path.join(
            "datasets",
            "experiment_ssd",
            "2018_07_25_14-00",
            "exported_model",
            "frozen_inference_graph.pb",
        )

        self.labelsArg = os.path.join("datasets", "records", "classes.pbtxt")

        self.num_classesArg = 37

        self.min_confidenceArg = 0.5

        self.numberPlateObj = DetectVehicleNumberPlate()


def decodeImageIntoBase64(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)

    with open(fileName, "wb") as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())


@app.route("/predict", methods=["POST"])
def getPrediction():
    inpImage = request.json["image"]

    decodeImageIntoBase64(inpImage, imagePath)

    try:
        labelledImage = clApp.numberPlateObj.predictImages(
            imagePath, pred_stagesArgVal, croppedImagepath, clApp.numberPlateObj
        )
        if labelledImage is not None:
            encodedCroppedImageStr = encodeImageIntoBase64(croppedImagepath)

            ig = str(encodedCroppedImageStr)

            ik = ig.replace("b'", "")

            numberPlateVal = detect_license_plate(ik)

            if len(numberPlateVal) == 10:
                responseDict = {"base64Image": ik, "numberPlateVal": numberPlateVal}

                jsonStr = json.dumps(responseDict, ensure_ascii=False).encode("utf8")

                return Response(jsonStr.decode())

            else:
                responseDict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}

                jsonStr = json.dumps(responseDict, ensure_ascii=False).encode("utf8")

                return Response(jsonStr.decode())
        else:
            responseDict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}

            jsonStr = json.dumps(responseDict, ensure_ascii=False).encode("utf8")

            return Response(jsonStr.decode())

    except Exception as e:
        print(e)

    responseDict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}

    jsonStr = json.dumps(responseDict, ensure_ascii=False).encode("utf8")

    return Response(jsonStr.decode())


if __name__ == "__main__":
    clApp = ClientApp()

    host = "127.0.0.1"

    port = 5000

    httpd = simple_server.make_server(host, port, app)

    print("Serving on %s %d" % (host, port))

    httpd.serve_forever()
