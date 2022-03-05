import boto3
import cv2
import numpy as np

class S3_Operation:
    def __init__(self):
        self.s3_resource = boto3.resource("s3")

    def get_bucket(self, bucket_name):
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

    def read_image(self, bucket_name, file_name):
        try:
            f_obj = self.get_file_object(file_name=file_name, bucket_name=bucket_name)

            file_stream = f_obj.get()["Body"].read()

            im = cv2.imdecode(np.asarray(bytearray(file_stream)), cv2.IMREAD_COLOR)

            return im

        except Exception as e:
            raise e

    def get_files_from_folder(self,folder_name,bucket_name):
        try:
            lst = self.get_file_object(bucket_name=bucket_name,file_name=folder_name)

            list_of_files = [object.key for object in lst]

            return list_of_files

        except Exception as e:
            raise e
    
    def read_images_from_folder(self,folder_name,bucket_name):
        try:
            lst_f = self.get_files_from_folder(folder_name=folder_name,bucket_name=bucket_name)

            extensions = ["jpeg","jpg","png"]
            
            img_lst = [self.read_image(bucket_name=bucket_name,file_name=f) for f in lst_f if f.split(".")[-1] in extensions]

            return img_lst            

        except Exception as e:
            raise e 


