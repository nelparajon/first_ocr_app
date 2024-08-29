import os
import sys
import cv2
import numpy as np
from pdf2image import convert_from_bytes, convert_from_path
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from process_pdf.ocr_producer import OCRProducer
from cliente.encoder import Encoder

class ProcessToImages:
    def __init__(self, output_dir='./output_images'):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    #para pruebas
    def convert_pdf_to_image_from_file(self, file):
        imgs = convert_from_path(file)
        images = {}
        for i,img in enumerate(imgs):
            images[i + 1] = img
        
        print(images)
        return images
    #para pruebas
    def convert_complete_pdf_to_image(self, file):
        imgs = convert_from_bytes(file)
        images = {}
        for i,img in enumerate(imgs):
            images[i + 1] = img
        
        print(images)
        return images

    #pruebas para las imagenes de los parrafos
    def save_single_image(self, image, name):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        cv2.imwrite(os.path.join(self.output_dir, f'{name}.png'), image)

    #se convierte en un array numpy para poder escalar a grises
    def image_to_nparray(self, image):
        print(f"Tipo de imagen recibido: {type(image)}")
        if isinstance(image, np.ndarray):
            return image
        
        elif isinstance(image, Image.Image):
            return np.array(image)
        
        else:
            try:
                return np.array(image)
            except Exception as e:
                raise ValueError(f"No se puede convertir la imagen a ndarray. Tipo no soportado: {type(image)}")

    #convierte la imagen a escala de grises
    def convert_to_gray(self, image):
        img = self.image_to_nparray(image)
        
        #verificamos si la imagen es valida, es decir, no es none o no tiene canales RGB
        if img is None or img.size == 0:
            raise ValueError("La imagen está vacía o es None, no se puede convertir a escala de grises.")
        #verificamos cuantos canales tiene y la convertimos a escala de grises cuando sea necesario (!=2)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        elif len(img.shape) == 2:
            img_gray = img
        else:
            raise ValueError("Formato de imagen no soportado.")
        
        return img_gray


    #threshold otsu automatico
    def thresh_otsu(self, image):
        _, img_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return img_otsu
    
    #invierte el binario de la imagen
    def invert_image(self, image):
        inverted_image = cv2.bitwise_not(image)
        return inverted_image
    
    #comprueba los contornos
    def search_contours(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    #añade blur a la imagen para procesarla
    def blur_image(self, image):
        img_gray = cv2.GaussianBlur(image, (7, 7), 0)
        return img_gray
    
    def kernel_image(self, image):
        img_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return img_k
    
    #hacemos que los tonos blanco en este caso se expandan para su posterior filtrado y procesamiento
    def dilate_image(self, image, kernel):
        img_dilate = cv2.dilate(image, kernel, iterations=4)
        return img_dilate
    
    #extraer solamente los parrafos como imagen delimitados por los contornos
    def extract_paragraph_images(self, image, contours, min_x=200, min_y=100, page_num=1):
        paragraph_images = []
        image_h, image_w = image.shape
        margen_x = 0.10 * image_w
        margen_y_top = 0.15 * image_h
        margen_y_bot = 0.15 * image_h
        image_with_rects = np.array(image)
        print(f"Image h: {image_h}\n Image w: {image_w}")
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            if x > margen_x and (x + w) < (image_w - margen_x) and y > margen_y_top and (y + h) < (image_h - margen_y_bot):
                if w > min_x and h > min_y:
                    roi = image[y:y+h, x:x+w]
                    cv2.rectangle(image_with_rects, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    paragraph_name = f'paragraph_page_{page_num}_{i}'
                    print(f"Imagen de párrafo guardada: {paragraph_name}")
                    self.save_single_image(roi, paragraph_name)
                    paragraph_images.append(roi)
                else:
                    print(f"Contorno {i} descartado por tamaño insuficiente (w={w}, h={h}).")
            else:
                print(f"Contorno {i} descartado por no estar en la zona central.")
                    
        return paragraph_images

    #filtra los contornos para extraer los parrafos correcto teniendo en cuenta, tamaño y posicion en la imagen
    def filter_paragraph_contours(self, image, contours, min_x=180, min_y=80):
        if isinstance(image, Image.Image):
            image = np.array(image)

        image_h, image_w = image.shape[:2]
        margen_x_left = 0.10 * image_w
        margen_x_right = 0.90 * image_w
        margen_y_top = 0.05 * image_h
        margen_y_bot = 0.95 * image_h
        
        valid_contours = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if x > margen_x_left and (x + w) < margen_x_right and y > margen_y_top and (y + h) < margen_y_bot:
                if w > min_x and h > min_y:
                    valid_contours.append((x, y, w, h))
                    print(f"Contorno válido: ({x}, {y}), ancho: {w}, alto: {h}")
                else:
                    print(f"Contorno descartado por tamaño insuficiente (w={w}, h={h}).")
            else:
                print(f"Contorno descartado por no estar en la zona central.")

        return valid_contours

    #funcion para añadir contornos a la imagen del documento
    #retorna la imagen además de las posiciones de los contornos para su posterior uso (quizás sea mejor separar ambas funcionalidades)
    def extract_image_with_contours(self, images):
        images_with_contours = {}
        contours_positions = {}

        for page_num, image in images.items():
            
            gray_img = self.convert_to_gray(image)
            blur_img = self.blur_image(gray_img)
            thresh_img = self.thresh_otsu(blur_img)
            inv_img = self.invert_image(thresh_img)
            kernel = self.kernel_image(inv_img)
            dilate = self.dilate_image(inv_img, kernel)

            contours = self.search_contours(dilate)
            
            valid_contours = self.filter_paragraph_contours(image, contours)
            
            if valid_contours:
                image_with_rects = np.array(image)
                positions = []

                for x, y, w, h in valid_contours:
                    cv2.rectangle(image_with_rects, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    #guardamos las posiciones de cada uno de los contornos que corresponden a los parrafos
                    positions.append({'x': x, 'y': y, 'width': w, 'height': h})

                images_with_contours[page_num] = image_with_rects
                contours_positions[page_num] = positions
                print("Imagenes y parrafos conseguidas con éxito")
            else:
                print(f"No se encontraron contornos válidos en la página {page_num}.")

        return images_with_contours, contours_positions


    
    
        

            