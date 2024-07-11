import nltk
from nltk import word_tokenize
from setup_nltk import setup_nltk
from nltk.corpus import stopwords
import string

class Tokenizer:

    def __init__(self):
        setup_nltk()  # Configura NLTK descargando los recursos necesarios
        self.stop_words = set(stopwords.words('english'))  # Usar un set para mejorar la eficiencia en la búsqueda

    def tokenize_texts(self, texts):
        tokenized_texts = []  # Lista para almacenar los textos tokenizados

        for text in texts:
            cleaned_text = self.clean_text(text)  # Limpiar el texto
            words = word_tokenize(cleaned_text)  # Tokenizar el texto limpio
            filtered_words = self.filter_words(words)  
            tokenized_texts.append(filtered_words)  
        print(tokenized_texts)
        print("Tokenización completada con éxito")
        return tokenized_texts
    
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
