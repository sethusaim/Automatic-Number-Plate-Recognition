
import boto3
from PIL import Image
import numpy as np

class S3_Operation:
    def __init__(self):
        self.s3_resource = boto3.resource("s3")

    def get_bucket(self,bucket_name):
        try:
            bucket = self.s3_resource.Bucket(bucket_name)

            return bucket

        except Exception as e:
            raise e 

    def get_file_object(self, file_name, bucket_name):
        try:
            bucket = self.get_bucket(bucket_name=bucket_name)

            lst_objs = [object for object in bucket.objects.filter(Prefix=file_name)]

            func = lambda x: x[0] if len(x) == 1 else x

            file_objs = func(lst_objs)

            return file_objs

        except Exception as e:
            raise e

    def read_image(self,bucket_name,file_name):
        try:
            f_obj = self.get_file_object(file_name=file_name,bucket_name=bucket_name)

            file_stream = f_obj.get()["Body"]

            im = Image.open(file_stream)

            return np.array(im)

        except Exception as e:
            raise e

s3 = S3_Operation()

image = s3.read_image(bucket_name="test-sethu",file_name="car3.jpeg")

print(image)
