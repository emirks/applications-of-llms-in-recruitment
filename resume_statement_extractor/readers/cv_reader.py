import cv2
from . import BaseReader

class CVReader(BaseReader):
    def read_image(self, image_path):
        return cv2.imread(image_path)