import json
import os
import re

import requests

url = os.environ["GCP_OCR_API"]

def detect_license_plate(encodedImage):
    headers = {"content-type": "application/json"}

    data = (
        """{
      "requests": [
        {
          "image": {
                   "content": '"""
        + encodedImage[:-1]
        + """'

                    },

          "features": [
            {
              "type": "TEXT_DETECTION"
            }
          ]
        }
      ]
    }"""
    )
    r = requests.post(url, headers=headers, data=data)

    result = json.loads(r.text)

    print(result)

    try:
        result = result["responses"][0]["textAnnotations"][0]["description"]

    except Exception as e:
        return r

    result = result.replace("\n", "").replace(" ", "")

    result = re.sub("\W+", "", result)

    mystates = [
        "AP",
        "AR",
        "AS",
        "BR",
        "CG",
        "GA",
        "GJ",
        "HR",
        "HP",
        "JK",
        "JH",
        "KA",
        "KL",
        "MP",
        "MH",
        "MN",
        "ML",
        "MZ",
        "NL",
        "OD",
        "PB",
        "RJ",
        "SK",
        "TN",
        "TS",
        "TR",
        "UA",
        "UK",
        "UP",
        "WB",
        "AN",
        "CH",
        "DN",
        "DD",
        "DL",
        "LD",
        "PY",
    ]

    if len(result) > 0:
        for word in mystates:
            if word in result:
                res = re.findall(
                    word + "[0-9]{1,2}\s*[A-Z]{1,2}\s*[0-9]{1,4}\s*]?", result
                )
                if len(res) > 0:
                    return res[0]
    return result
