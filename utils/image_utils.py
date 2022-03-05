import base64


class Image_Utils:
    def __init__(self):
        self.class_name = self.__class__.__name__

    def decode_image(self,imgstring, file_name):
        method_name = self.decode_image.__name__

        try:
            imgdata = base64.b64decode(imgstring)

            with open(file_name, "wb") as f:
                f.write(imgdata)

                f.close()

        except Exception as e:
            raise Exception(f"Exception occured in Class : {self.class_name}, Method : {method_name}, Error : {str(e)}")


    def encode_image(self,cropped_img_path):
        method_name = self.encode_image.__name__

        try:
            with open(cropped_img_path, "rb") as f:
                return base64.b64encode(f.read())

        except Exception as e:
            raise Exception(f"Exception occured in {__file__}, Method : {method_name}, Error : {str(e)}")
