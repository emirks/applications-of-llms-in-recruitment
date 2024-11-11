from . import BaseParser
import pytesseract
import cv2

class TesseractParser(BaseParser):
    def preprocess_images(self, images):
        preprocessed_images = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            preprocessed_images.append(thresh)
        return preprocessed_images

    def parse_text(self, pdf_file, images):
        texts = ""
        for image in images:
            custom_config = r'--oem 1 --psm 6 -l eng -c preserve_interword_spaces=1 -c load_system_dawg=1 -c load_freq_dawg=1'
            text = pytesseract.image_to_string(image, config=custom_config)
            texts += text + "\n"
        return texts
