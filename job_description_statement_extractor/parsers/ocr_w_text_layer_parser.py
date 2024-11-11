import io
from . import BaseParser
import pytesseract
import cv2
from pdfminer.high_level import extract_text


class OcrWTextLayerParser(BaseParser):
    def preprocess_images(self, images):
        preprocessed_images = []
        for image in images:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            preprocessed_images.append(thresh)
        return preprocessed_images
    
    def clean_text(self, text:str):
        lines = text.split('\n')        
        cleaned_lines = [line.strip() for line in lines if line.strip()]
        cleaned_text = '\n'.join(cleaned_lines)
        return cleaned_text

    def parse_text(self, pdf_file, images):
        # # get text from the pdf from its text layer
        # pdf_text = ""
        # with fitz.open(pdf_path) as pdf:
        #     for page_num in range(len(pdf)):
        #         page = pdf.load_page(page_num)
        #         pdf_text += page.get_text()

        if isinstance(pdf_file, bytes):
            pdf_file_like = io.BytesIO(pdf_file)
        else:
            pdf_file_like = pdf_file
        pdf_text = extract_text(pdf_file_like)

        texts = ""
        for image in images:
            custom_config = r'--oem 1 --psm 6 -l eng -c preserve_interword_spaces=1 -c load_system_dawg=1 -c load_freq_dawg=1'
            text = pytesseract.image_to_string(image, config=custom_config)
            texts += text + "\n"

        combined_text = f"""
        PDF Layer Text:
        {pdf_text}

        OCR Text:
        {texts}
        """
        return self.clean_text(combined_text)
    
    def parse_text_to_json(self, pdf_file, images):
        if isinstance(pdf_file, bytes):
            pdf_file_like = io.BytesIO(pdf_file)
        else:
            pdf_file_like = pdf_file
        pdf_text = extract_text(pdf_file_like)
        
        texts = ""
        for image in images:
            custom_config = r'--oem 1 --psm 6 -l eng -c preserve_interword_spaces=1 -c load_system_dawg=1 -c load_freq_dawg=1'
            text = pytesseract.image_to_string(image, config=custom_config)
            texts += text + "\n"

        parsed_text = {
            "pdf_text": self.clean_text(pdf_text),
            "ocr_text": self.clean_text(texts)
        }
        return parsed_text