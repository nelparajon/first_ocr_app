import nltk
from nltk import word_tokenize
from setup_nltk import setup_nltk
from nltk.corpus import stopwords
import string

class Tokenizer:

    def __init__(self):
        setup_nltk()
        self.tokenized_texts = []
        self.stop_words = stopwords.words('spanish')

    def tokenizer_text(self, texts):
        self.tokenized_texts = []

        for text in texts:
            cleaned_text = self.clean_text(text)
            words = word_tokenize(cleaned_text)
            filtered_words = self.filter_words(words)
            self.tokenized_texts.append(filtered_words)
            print("************TEXTOS FILTRADOS*************")
            print(filtered_words)

        print("Tokenización completada con éxito")
        return self.tokenized_texts
    
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
