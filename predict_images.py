import cv2
import tensorflow as tf
from PIL import Image

from base2designs.plates.plateDisplay import PlateDisplay
from base2designs.plates.plateFinder import PlateFinder
from base2designs.plates.predicter import Predicter
from base2designs.utils import label_map_util


class DetectVehicleNumberPlate:
    def __init__(self):
        self.modelArg = "datasets/experiment_ssd/2018_07_25_14-00/exported_model/frozen_inference_graph.pb"

        self.labelsArg = "datasets/records/classes.pbtxt"

        self.num_classesArg = 37

        self.min_confidenceArg = 0.5

        self.model = tf.Graph()

        with self.model.as_default():
            self.graphDef = tf.GraphDef()

            with tf.gfile.GFile(self.modelArg, "rb") as f:
                self.serializedGraph = f.read()

                self.graphDef.ParseFromString(self.serializedGraph)

                tf.import_graph_def(self.graphDef, name="")

        self.labelMap = label_map_util.load_labelmap(self.labelsArg)

        self.categories = label_map_util.convert_label_map_to_categories(
            self.labelMap, max_num_classes=self.num_classesArg, use_display_name=True
        )

        self.categoryIdx = label_map_util.create_category_index(self.categories)

        self.plateFinder = PlateFinder(
            self.min_confidenceArg, self.categoryIdx, rejectPlates=False, charIOUMax=0.3
        )

        self.plateDisplay = PlateDisplay()

    def predictImages(
        self, imagePathArg, pred_stagesArg, croppedImagepath, numPlateOrg
    ):
        with numPlateOrg.model.as_default():
            with tf.Session(graph=numPlateOrg.model) as sess:
                predicter = Predicter(numPlateOrg.model, sess, numPlateOrg.categoryIdx)

                image = cv2.imread(imagePathArg)

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
                        image, plateBoxes_pred, imagePathArg, croppedImagepath
                    )

                else:
                    print(
                        "[ERROR] --pred_stages {}. The number of prediction stages must be either 1 or 2".format(
                            pred_stagesArg
                        )
                    )
                    quit()

                return imageLabelled

    def getBoundingBox(self, image, plateBoxes, imagePath, croppedImagepath):
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

                cropped_image.save(croppedImagepath)

                return cropped_image

            except Exception as e:
                raise e
