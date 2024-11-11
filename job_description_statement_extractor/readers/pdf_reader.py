from pdf2image import convert_from_path, convert_from_bytes
from . import BaseReader

class PDFReader(BaseReader):
    def read_image(self, pdf_file, dpi=100):
        try: 
            if isinstance(pdf_file, bytes):
                pdf_images = convert_from_bytes(pdf_file, dpi)
            else:
                pdf_images = convert_from_path(pdf_file, dpi)
            return pdf_images
        except Exception as e:
            print(e)
            return None