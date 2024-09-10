import os
import sys
import cv2
import numpy as np

import pytesseract

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


    def save_images(self, images):
        for i, image in enumerate(images.items()):
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            cv2.imwrite(os.path.join(self.output_dir, f'{i}'))



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

        try:
            if image is None:
                raise ValueError("La imagen proporcionada es None.")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
            _, img_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img_otsu
        except cv2.error as e:
            print(f"Error de OpenCV durante la umbralización: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error inesperado: {ex}")
            raise
    
    #invierte el binario de la imagen
    def invert_image(self, image):
        try:
            if image is None:
                raise ValueError("No existe ninguna imagen para invertir su binario")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            inverted_image = cv2.bitwise_not(image)
            return inverted_image
        except cv2.error as e:
            print(f"error de opencv durante el proceso de inversión del binario de la imagen: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto durante el proceso de inversión del binario de la imagen:{ve}")
            raise
        except TypeError as te:
            print(f"Tipo de dato incorrecto durante el proceso de inversión del binario de la imagen:{te}")
            raise
        except Exception as ex:
            print(f"Error inesperado durante el proceso de inversión del binario de la imagen: {ex}")
    
    #comprueba los contornos
    def search_contours(self, image):
        try:
            if image is None:
                raise ValueError("La imagen proporcionada no existe.")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
            if len(image.shape) != 2:
                raise ValueError("La imagen proporcionada no está en escala de grises.")
            
            contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return contours
        except cv2.error as e:
            print(f"Error de OpenCV durante la búsqueda de contornos: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto durante la busqueda de contornos: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto durante la busqueda de contornos: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error durante la busqueda de contornos: {ex}")
            raise
    #añade blur a la imagen para procesarla
    def blur_image(self, image):
        try:
            if image is None:
                raise ValueError("La imagen proporcionada no existe.")
            if not isinstance(image, (np.ndarray,)):
                raise TypeError("El tipo de dato de la imagen no es válido. Se esperaba un numpy.ndarray.")
            
           #comprobar que la imagen esté en escala de grises (2 canales) o en color (3 canales)
            if len(image.shape) < 2 or len(image.shape) > 3:
                raise ValueError("La imagen proporcionada tiene un número de dimensiones incorrecto.")
            
            img_gray = cv2.GaussianBlur(image, (7, 7), 0)
            return img_gray
        except cv2.error as e:
            print(f"Error de OpenCV durante el desenfoque de la imagen: {e}")
            raise
        except ValueError as ve:
            print(f"Valor incorrecto durante el desenfoque de la imagen: {ve}")
            raise
        except TypeError as te:
            print(f"Tipo incorrecto durante el desenfoque de la imagen: {te}")
            raise
        except Exception as ex:
            print(f"Ocurrió un error inesperado durante el desenfoque de la imagen: {ex}")
            raise

    
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


    
    #hacemos un filtrado de los contornos para descartar cuáles son títulos y cuáles no
    #para esto usamos el tamaño altura y ancho. Un título en principio será mas ancho que alto
    #además, pueden estar en negrita por tanto ajustamos un threshold y una variable para el calculo de la densidad y las comparamos
    def filter_title_contours(self, image, contours, min_width_ratio=0.15, max_height_ratio=0.07, density_threshold=0.25):
        """
        Filtra los contornos que corresponden a títulos según su tamaño relativo y densidad de píxeles negros.
        Se ha ajustado el ratio de ancho y altura para detectar mejor los títulos.
        
        Args:
            image: La imagen original en la que se buscan los títulos.
            contours: Los contornos detectados en la imagen procesada.
            min_width_ratio: Relación mínima de ancho en función del ancho total de la imagen.
            max_height_ratio: Relación máxima de altura en función de la altura total de la imagen.
            density_threshold: Umbral de densidad de píxeles negros para identificar títulos.
        
        Returns:
            valid_titles: Lista de contornos que corresponden a títulos válidos.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        image_h, image_w = image.shape[:2]
        print(f"Altura de la imagen: {image_h}, Ancho de la imagen: {image_w}")

        valid_titles = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Contorno encontrado en ({x}, {y}), ancho: {w}, alto: {h}")
            
            # Condiciones mejoradas para detectar los títulos
            if w > min_width_ratio * image_w and h < max_height_ratio * image_h:
                # Extraer la región del contorno
                region_contour = image[y:y+h, x:x+w]
                # Calcular la densidad de píxeles negros (0 representa negro en imágenes binarizadas)
                pixel_density = np.sum(region_contour == 0) / (w * h)
                print(f"Densidad de píxeles: {pixel_density}")
                
                # Filtrar por densidad de píxeles negros
                if pixel_density > density_threshold:
                    valid_titles.append((x, y, w, h))
                    print(f"Contorno válido como título: ({x}, {y}), ancho: {w}, alto: {h}")
                else:
                    print(f"Contorno descartado por baja densidad de píxeles: {pixel_density}")
            else:
                print(f"Contorno descartado por no cumplir con las medidas relativas (w={w}, h={h}).")

        return valid_titles
    
    def extraer_titulos(self, images):
        """
        Procesa cada página de la lista de imágenes para extraer las posiciones de los títulos.
        """
        image_with_titles = {}
        titles_position = {}

        for page_num, image in images.items():
            gray_img = self.convert_to_gray(image)
            blur_img = self.blur_image(gray_img)
            thresh_img = self.thresh_otsu(blur_img)
            inv_img = self.invert_image(thresh_img)
            kernel = self.kernel_image(inv_img)
            dilate = self.dilate_image(inv_img, kernel)

            # Buscar contornos en la imagen dilatada
            contours = self.search_contours(dilate)
            
            # Filtrar los contornos que corresponden a títulos válidos
            valid_titles = self.filter_title_contours(image, contours)

            if valid_titles:
                image_with_titles_page = np.array(image)
                positions = []

                # Dibujar los rectángulos en los contornos de títulos
                for x, y, w, h in valid_titles:
                    cv2.rectangle(image_with_titles_page, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    positions.append({'x': x, 'y': y, 'width': w, 'height': h})
                    self.save_single_image(image_with_titles_page, f"page{page_num}")

                # Guardar las imágenes y posiciones de títulos
                image_with_titles[page_num] = image_with_titles_page
                titles_position[page_num] = positions
                print("Títulos extraídos con éxito")
            else:
                print(f"No se encontraron contornos válidos en la página {page_num}.")
        
        print(titles_position)
        
        return image_with_titles, titles_position
    
    

    def extraer_texto_titulos(self, images, titles_positions):
        titulos_texto = {}
        last_title_id = None
        print(images)

        for page_num, positions in titles_positions.items():
            image = images[page_num]
            image_height, image_width = image.shape[:2]  #tamaño de la imagen

            # Ordenar las posiciones de los títulos según la coordenada 'y' (de arriba hacia abajo)
            positions_sorted = sorted(positions, key=lambda pos: pos['y'])

            for i, pos in enumerate(positions_sorted):
                x, y, w, h = pos['x'], pos['y'], pos['width'], pos['height']
                #recortamos la imagen donde se encuentra cada titulo
                titulo_img = image[y:y+h, x:x+w]
                
                #aplicamos ocr
                titulo_txt = ocr.img_to_txt(titulo_img)

                #identificador para los titulos del diccionario
                titulo_id = f'page_{page_num}_titulo_{i+1}'

                #guardamos los titulos en un diccionario
                titulos_texto[titulo_id] = {
                    'texto': titulo_txt,
                    'parrafos': "",
                    'coordenadas': {'x': x, 'y': y, 'width': w, 'height': h}
                }

                #si hay un siguiente título en la misma página, extraer el contenido entre títulos
                if i < len(positions_sorted) - 1:
                    next_pos = positions_sorted[i + 1]  #extraemos sus coordenadas del diccionario de posiciones
                    y2 = next_pos['y']  #la Y corresponde al límite del siguiente título

                    #recortamos el contenido entre título y título
                    contenido_img = image[y+h:y2, 0:image_width]

                    if contenido_img.size > 0: #el tamaño debe ser mayor que cero para que sea válido
                        contenido_txt = ocr.img_to_txt(contenido_img)
                        contenido_txt.strip()
                        titulos_texto[titulo_id]['parrafos'] = contenido_txt #añadimos el contenido al diccionario
                        print(f"Contenido entre títulos en página {page_num}: {contenido_txt}")
                else:
                    #si es el último título de la página, extraemos hasta el final de la página
                    contenido_img = image[y+h:image_height, 0:image_width]
                    if contenido_img.size > 0:
                        contenido_txt = ocr.img_to_txt(contenido_img)
                        contenido_txt.strip()
                        titulos_texto[titulo_id]['parrafos'] = contenido_txt
                        print(f"Contenido hasta el final de la página {page_num}: {contenido_txt}")

                    #guardamos el último título para concatenar si el contenido continúa en la siguiente página
                    last_title_id = titulo_id
            """ 
                Esta parte no me acaba de funcionar correctamente. 
                Añade como valor al título anterior tanto la parte correspondiente de los parrafos de la siguiente página
                cómo también el primer título y su contenido.
                Creo que es por un error en el recorte, pero no he dado con ello 
            """
            #si hay una siguiente página y el contenido continúa (sin título al inicio de la página siguiente)
            if last_title_id and (page_num + 1) in images:
                next_page_image = images[page_num + 1] #extraermos la imagen correspondiente de la página 
                next_positions = titles_positions.get(page_num + 1, []) #extraemos las posiciones de los titulos

                if next_positions:
                    #si la siguiente página tiene títulos, cortar el contenido hasta el primer título de la página siguiente
                    first_title_next_page = next_positions[0] #se supone que la primera posición es el primer título
                    y_next_first_title = first_title_next_page['y'] #este es el límite del primer título

                    contenido_img_next_page = next_page_image[0:y_next_first_title, 0:image_width]

                    if contenido_img_next_page.size > 0:
                        contenido_txt_next_page = ocr.img_to_txt(contenido_img_next_page)
                        contenido_txt_next_page.strip()
                        titulos_texto[last_title_id]['parrafos'] += "\n" + contenido_txt_next_page
                        print(f"Contenido añadido al título '{titulos_texto[last_title_id]['texto'].strip()}' desde la página siguiente: {contenido_txt_next_page}")
                else:
                    #si no hay títulos en la página siguiente, cortamos toda la página hasta el final
                    contenido_img_next_page = next_page_image[0:image_height, 0:image_width]
                    if contenido_img_next_page.size > 0:
                        contenido_txt_next_page = ocr.img_to_txt(contenido_img_next_page)
                        contenido_txt_next_page.strip()
                        titulos_texto[last_title_id]['parrafos'] += "\n" + contenido_txt_next_page
                        print(f"Todo el contenido de la página {page_num + 1} añadido al título '{titulos_texto[last_title_id]['texto'].strip()}': {contenido_txt_next_page}")

                #reseteamos el título que se usa para verificar títulos en la página anterior y añadir el contenido de la siguiente
                last_title_id = None

        return titulos_texto



if __name__ == '__main__':
    pti = ProcessToImages()
    ocr = OCRProducer()
    doc1 = './docs_pruebas/lematizacion.pdf'
    images = pti.convert_pdf_to_image_from_file(doc1)
    image_parrafos, positions = pti.extract_image_with_contours(images)
    

    images, titulos = pti.extraer_titulos(images)
    titulos_y_parrafos_images = pti.extraer_texto_titulos(images, titulos)
    print(titulos_y_parrafos_images)

