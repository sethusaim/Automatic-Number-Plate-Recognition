import cv2
from PIL import Image


def crop(image_path, coords, saved_location):
    try:
        image_obj = Image.open(image_path)

        cropped_image = image_obj.crop(coords)
        cropped_image = cropped_image.convert("RGB")
        cropped_image.save(saved_location)
        return cropped_image
    except Exception as e:
        print(e)
    # cropped_image.show()


class Plate_Display:

    # create an annotated image with plate boxes, char boxes, and labels
    def labelImage(self, image, plate_boxes, image_path, cropped_img_path):
        (H, W) = image.shape[:2]

        for plateBox in plate_boxes:
            # Draw the plate box rectangle in red
            # scale the bounding box from the range [0, 1] to [W, H]
            (startY, startX, endY, endX) = plateBox
            startX = int(startX * W)
            startY = int(startY * H)
            endX = int(endX * W)
            endY = int(endY * H)
            croppedimage = crop(
                image_path, (startX, startY, endX, endY), cropped_img_path
            )

        # loop over the plate text predictions
        # for (plateBox, chBoxes, charText) in zip(plate_boxes, charBoxes, charTexts):
        #     # Draw the plate box rectangle in red
        #     # scale the bounding box from the range [0, 1] to [W, H]
        #     (startY, startX, endY, endX) = plateBox
        #     startX = int(startX * W)
        #     startY = int(startY * H)
        #     endX = int(endX * W)
        #     endY = int(endY * H)
        #     # draw the plate box on the output image
        #     # cv2.rectangle(image, (startX, startY), (endX, endY),
        #     #               (0, 0, 255), 1)
        #
        #     crop(image_path, (startX, startY, endX, endY), cropped_img_path)
        #     # # Draw the char boxes and text labels in green
        #     # for (chBox, char) in zip(chBoxes, charText):
        #     #     (startY, startX, endY, endX) = chBox
        #     #     startX = int(startX * W)
        #     #     startY = int(startY * H)
        #     #     endX = int(endX * W)
        #     #     endY = int(endY * H)
        #     #     # draw the char box and label on the output image
        #     #     cv2.rectangle(image, (startX, startY), (endX, endY),
        #     #                   (0, 255, 0), 1)
        #     #     y = startY - 10 if startY - 10 > 10 else startY + 10
        #     #     cv2.putText(image, char, (startX, y),
        #     #                 cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        return croppedimage

    # def labelImage(self, image, plate_boxes):
    #     (H, W) = image.shape[:2]
    #     # loop over the plate text predictions
    #     for (plateBox) in zip(plate_boxes):
    #         # Draw the plate box rectangle in red
    #         # scale the bounding box from the range [0, 1] to [W, H]
    #         # plateBox = plateBox.split(sep=",")
    #         (startY, startX, endY, endX) = plateBox
    #         startX = int(startX * W)
    #         startY = int(startY * H)
    #         endX = int(endX * W)
    #         endY = int(endY * H)
    #         # draw the plate box on the output image
    #         cv2.rectangle(image, (startX, startY), (endX, endY),
    #                       (0, 0, 255), 1)
    #
    #         # # Draw the char boxes and text labels in green
    #         # for (chBox, char) in zip(chBoxes, charText):
    #         #   (startY, startX, endY, endX) = chBox
    #         #   startX = int(startX * W)
    #         #   startY = int(startY * H)
    #         #   endX = int(endX * W)
    #         #   endY = int(endY * H)
    #         #   # draw the char box and label on the output image
    #         #   cv2.rectangle(image, (startX, startY), (endX, endY),
    #         #                 (0, 255, 0), 1)
    #         #   y = startY - 10 if startY - 10 > 10 else startY + 10
    #         #   cv2.putText(image, char, (startX, y),
    #         #               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    #
    #     return image
