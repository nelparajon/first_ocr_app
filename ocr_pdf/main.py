import logging
from pdf_converter import PDFConverter
from ocr_producer import OCRProducer
from analyze_text import AnalyzeText
from tokenizer import Tokenizer

def menu(route_path):
    try:
        print('********************')
        print("Convirtiendo los pdf a imágenes...")
        converter = PDFConverter(route_path)
        images = converter.convert_to_images(route_path)

        print('****************')
        print('Procesando imágenes con OCR pytesseract...')
        ocr = OCRProducer()
        result_text = ocr.process_images(images)

        print('********************')
        print("Analizando los textos como imagen...")
        ocr.save_texts(result_text)
        analyzer = AnalyzeText()
        analyzer.count_words(result_text)
        words = analyzer.words_used(result_text)
        analyzer.words_most_used(words)
        analyzer.most_used_word_in_each_text(result_text)

        print("OCR completado con éxito")
    except Exception as e:
        logging.error(f"Error durante el procesamiento de los pdfs {e}")
        raise #lanzar la excepción de nuevo más allá del bloque except

    print("Proceso de Tokenización")
    tokenizer = Tokenizer()
    tokenizer.tokenizer_text(result_text)
   
if __name__ == '__main__':
    route_path = r'C:/Users/Nel/Desktop/carta de presentación nel parajon somoano.pdf'
    menu(route_path)