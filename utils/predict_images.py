import os

import cv2
import tensorflow as tf
from base2designs.plates.plateDisplay import Plate_Display
from base2designs.plates.plateFinder import Plate_Finder
from base2designs.plates.predicter import Predicter
from base2designs.utils import label_map_util
from PIL import Image


class DetectVehicleNumberPlate:
    def __init__(self):
        self.model_arg = os.path.join(
            "datasets",
            "experiment_ssd",
            "2018_07_25_14-00",
            "exported_model",
            "frozen_inference_graph.pb",
        )

        self.labels_arg = os.path.join("datasets", "records", "classes.pbtxt")

        self.num_classes_arg = 37

        self.min_confidenceArg = 0.5

        self.model = tf.Graph()

        with self.model.as_default():
            self.graphDef = tf.GraphDef()

            with tf.gfile.GFile(self.model_arg, "rb") as f:
                self.serializedGraph = f.read()

                self.graphDef.ParseFromString(self.serializedGraph)

                tf.import_graph_def(self.graphDef, name="")

        self.labelMap = label_map_util.load_labelmap(self.labels_arg)

        self.categories = label_map_util.convert_label_map_to_categories(
            self.labelMap, max_num_classes=self.num_classes_arg, use_display_name=True
        )

        self.categoryIdx = label_map_util.create_category_index(self.categories)

        self.plateFinder = Plate_Finder(
            self.min_confidenceArg, self.categoryIdx, rejectPlates=False, charIOUMax=0.3
        )

        self.plateDisplay = Plate_Display()

    def predictImages(self, image_path, pred_stagesArg, cropped_img_path, num_plate_org):
        try:
            with num_plate_org.model.as_default():
                with tf.Session(graph=num_plate_org.model) as sess:
                    predicter = Predicter(num_plate_org.model, sess, num_plate_org.categoryIdx)

                    image = cv2.imread(image_path)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if pred_stagesArg == 2:
                        boxes, scores, labels = predicter.predictPlates(
                            image, preprocess=False
                        )
                        (
                            licensePlateFound_pred,
                            plateBoxes_pred,
                            plateScores_pred,
                        ) = self.plateFinder.findPlatesOnly(boxes, scores, labels)
                        imageLabelled = self.getBoundingBox(
                            image, plateBoxes_pred, image_path, cropped_img_path
                        )

                    else:
                        print(
                            "[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(
                                pred_stagesArg
                            )
                        )
                        quit()

                    return imageLabelled

        except Exception as e:
            raise e
    def getBoundingBox(self, image, plateBoxes, image_path, cropped_img_path):
        (H, W) = image.shape[:2]

        for plateBox in plateBoxes:
            (startY, startX, endY, endX) = plateBox

            startX = int(startX * W)

            startY = int(startY * H)

            endX = int(endX * W)

            endY = int(endY * H)

            try:
                image_obj = Image.open(image_path)

                cropped_image = image_obj.crop((startX, startY, endX, endY))

                cropped_image = cropped_image.convert("L")

                cropped_image.save(cropped_img_path)

                return cropped_image

            except Exception as e:
                raise e
