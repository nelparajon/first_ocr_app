from process_pdf.pdf_converter import PDFConverter
from process_pdf.ocr_producer import OCRProducer
from process_pdf.tokenizer import Tokenizer
from process_pdf.stemmer import Stemmer
from process_pdf.lemmatizer import Lemmatizer
from process_pdf.output_analyzed_texts import AnalyzedText
from process_pdf.vectorizer import Vectorizer
from setup_nltk import setup_nltk

def documents(*file_paths):
    docs = []
    for file_path in file_paths:
        docs.append(file_path)
    print(docs)
    return docs

def convert_to_images_from_file(docs):
    print('********************')
    print("Convirtiendo los pdf a imágenes...")
    images = []
    for route in docs:
        converter = PDFConverter(route)
        docs_images = converter.convert_to_images_from_file(route)
        images.append(docs_images)
    
    return images

def process_images(images):
    result_texts = []
    for image in images:
        ocr_producer = OCRProducer()
        texts = ocr_producer.process_images(image)
        result_texts.append(texts)
    
    return result_texts

def tokenize_texts(texts):
    tokenizer = Tokenizer()
    tokens = []
    for text in texts:
        
        text_tokens = tokenizer.tokenize_texts(text)
        tokens.append(text_tokens)
    return tokens


def lemmatizing_texts(tokens):
    lemmatizer = Lemmatizer()
    lematized_tokens = []
    for text in tokens:
        lem_tokens = lemmatizer.lemmatizing_words(text)
        
        lematized_tokens.append(lem_tokens)
    print(lematized_tokens)
    return lematized_tokens

def vectorizing_texts(lems):
    vectorizer = Vectorizer()
    matrix = vectorizer.vectorize_doc(lems)
    return matrix


def menu(docs):
    pass


if __name__ == '__main__':
    setup_nltk()
    doc_1 = r'C:/Users/Nel/Desktop/vectorizacion.pdf'
    doc_2 = r'C:/Users/Nel/Desktop/vectorization_2.pdf'
    docs = documents(doc_1, doc_2) 
    print("rutas de los documentos: ", docs)
    images = convert_to_images_from_file(docs)
    result_text = process_images(images)
    print(result_text)
    print("\n", type(result_text))
    tokenized_texts =  tokenize_texts(result_text)
    lem_texts = lemmatizing_texts(tokenized_texts)
    vectorizer = Vectorizer()
    vectors = vectorizing_texts(lem_texts)
    print("Vectores: \n", vectors)
    vectorizer.similarity_docs(vectors)