import cv2
import tensorflow as tf
from base2designs.plates.plateDisplay import Plate_Display
from base2designs.plates.plateFinder import Plate_Finder
from base2designs.plates.predicter import Predicter
from base2designs.utils.label_map_util import (convert_label_map_to_categories,
                                               create_category_index,
                                               load_labelmap)
from PIL import Image

from utils.read_params import read_params


class DetectVehicleNumberPlate:
    def __init__(self):
        self.config = read_params()

        self.model_arg = self.config["exported_model_path"]

        self.labels_arg = "datasets/records/classes.pbtxt"

        self.labels_arg = self.config["labels_file"]

        self.num_classes_arg = self.config["num_classes"]

        self.min_confidence_arg = self.config["min_confidence"]

        self.model = tf.Graph()

        with self.model.as_default():
            self.graphDef = tf.GraphDef()

            with tf.gfile.GFile(self.model_arg, "rb") as f:
                self.serializedGraph = f.read()

                self.graphDef.ParseFromString(self.serializedGraph)

                tf.import_graph_def(self.graphDef, name="")

        self.labelMap = load_labelmap(self.labels_arg)

        self.categories = convert_label_map_to_categories(
            self.labelMap, max_num_classes=self.num_classes_arg, use_display_name=True
        )

        self.categoryIdx = create_category_index(self.categories)

        self.plateFinder = Plate_Finder(
            self.min_confidence_arg,
            self.categoryIdx,
            rejectPlates=False,
            charIOUMax=0.3,
        )

        self.plateDisplay = Plate_Display()

    def predict_images(
        self, image_path_arg, pred_stages_arg, cropped_image_path, num_plate_org
    ):
        try:
            with num_plate_org.model.as_default():
                with tf.Session(graph=num_plate_org.model) as sess:
                    predicter = Predicter(
                        num_plate_org.model, sess, num_plate_org.categoryIdx
                    )

                    image = cv2.imread(image_path_arg)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    if pred_stages_arg == 2:
                        boxes, scores, labels = predicter.predict_plates(
                            image, preprocess=False
                        )

                        _, plates_boxes_pred, _ = self.plateFinder.find_plates_only(
                            boxes, scores, labels
                        )

                        labelled_image = self.get_bounding_box(
                            image, plates_boxes_pred, image_path_arg, cropped_image_path
                        )

                    else:
                        print(
                            f"[ERROR] --pred_stages {pred_stages_arg}.The number of prediction stages must be either 1 or 2"
                        )

                        quit()

                    return labelled_image

        except Exception as e:
            raise e

    def get_bounding_box(self, image, plateBoxes, imagePath, cropped_image_path):
        try:
            (H, W) = image.shape[:2]

            for plateBox in plateBoxes:
                (startY, startX, endY, endX) = plateBox

                startX = int(startX * W)

                startY = int(startY * H)

                endX = int(endX * W)

                endY = int(endY * H)

                try:
                    image_obj = Image.open(imagePath)

                    cropped_image = image_obj.crop((startX, startY, endX, endY))

                    cropped_image = cropped_image.convert("L")

                    cropped_image.save(cropped_image_path)

                    return cropped_image

                except Exception as e:
                    print(e)

        except Exception as e:
            raise e
