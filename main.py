import json
import os
from wsgiref import simple_server

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS, cross_origin

from utils.getNumberPlateVals import detect_license_plate
from utils.image_utils import Image_Utils
from utils.predict_images import DetectVehicleNumberPlate
from utils.read_params import read_params

app = Flask(__name__)

os.putenv("LANG", "en_US.UTF-8")
os.putenv("LC_ALL", "en_US.UTF-8")

CORS(app)

config = read_params()


@app.route("/", methods=["GET"])
@cross_origin()
def home():
    return render_template(config["templates"]["index"])


@app.route("/predict", methods=["POST"])
def getPrediction():
    inpImage = request.json["image"]

    img_utils = Image_Utils()

    img_utils.decode_image(inpImage, config["input_image_path"])

    try:
        num_plate = DetectVehicleNumberPlate()

        labelled_image = num_plate.predictImages(
            image_path=config["input_image_path"],
            pred_stagesArg=config["pred_stages_val"],
            cropped_img_path=config["results_path"],
            numPlateOrg=num_plate,
        )

        if labelled_image is not None:
            cropped_img_str = img_utils.encode_image(
                cropped_img_path=config["results_path"]
            )

            ig = str(cropped_img_str)

            ik = ig.replace("b'", "")

            numberPlateVal = detect_license_plate(ik)

            if len(numberPlateVal) == 10:
                response_dict = {"base64Image": ik, "numberPlateVal": numberPlateVal}

                jsonStr = json.dumps(response_dict, ensure_ascii=False).encode("utf8")

                return jsonify([{"numberPlateVal": numberPlateVal}])

            else:
                response_dict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}

                jsonStr = json.dumps(response_dict, ensure_ascii=False).encode("utf8")

                return jsonify(jsonStr.decode())
        else:
            response_dict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}

            jsonStr = json.dumps(response_dict, ensure_ascii=False).encode("utf8")

            return jsonify(jsonStr)

    except Exception as e:
        print(e)

    response_dict = {"base64Image": "Unknown", "numberPlateVal": "Unknown"}

    jsonStr = json.dumps(response_dict, ensure_ascii=False).encode("utf8")

    return jsonify(jsonStr)


if __name__ == "__main__":
    host = "0.0.0.0"

    port = 8080

    httpd = simple_server.make_server(host, port, app)

    print("Serving on %s:%d" % (host, port))

    httpd.serve_forever()
