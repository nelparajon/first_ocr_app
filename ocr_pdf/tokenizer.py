import os
import sys
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from setup_nltk import setup_nltk

class Tokenizer:

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))  # Usar un set para mejorar la eficiencia en la búsqueda, no admite duplicados

    def tokenize_texts(self, texts):

        try:
             #Verificar que el tipo de dato pasado sea una cadena
            if not isinstance(texts, list) and not all(isinstance(item, str) for item in texts):
                raise TypeError(f"Se esperaba una lista de cadenas de texto para tokenizar, pero se recibió {type(texts).__name__}")
            
            tokenized_texts = []  #Lista para almacenar todas las palabras tokenizadas
            cleaned_text = self.clean_text(texts) 
            
            if not cleaned_text:
                raise ValueError("El texto no está limpio de signos de puntuación, caracteres especiales y mayúsculas. Tokenización")

            try:
                words = word_tokenize(cleaned_text)  #Tokenizar el texto limpio
            except Exception as e:
                raise RuntimeError(f"Error al tokenizar el texto: {e}")

            try:
                filtered_words = self.filter_words(words)  
            except Exception as e:
                raise RuntimeError(f"Error al filtrar las palabras para tokenizar: {e}")

            tokenized_texts.extend(filtered_words)  #Añadir las palabras filtradas a la lista principal que se retorna
            print(tokenized_texts)
            print("Tokenización completada con éxito")
            return tokenized_texts
        
        except TypeError as te:
            print(f"TypeError: {te}")  #Errores de tipo
        except ValueError as ve:
            print(f"ValueError: {ve}") #Errores de valor
        except RuntimeError as re:
            print(f"RuntimeError: {re}") #Errores en tiempo de ejecución
        except Exception as e:
            print(f"Error inesperado en la tokenización: {e}") #Otros errores
            return []
    
    def filter_words(self, words):
        filtered_words = []
        for word in words:
            if word not in self.stop_words:
                filtered_words.append(word)
        return filtered_words

    def clean_text(self, text):
        text = text.lower()
        cleaned_text = text.translate(str.maketrans("", "", string.punctuation))
        return cleaned_text