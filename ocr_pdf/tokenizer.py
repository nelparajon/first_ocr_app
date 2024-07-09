import nltk
from nltk import word_tokenize
from setup_nltk import setup_nltk
from nltk.corpus import stopwords

class Tokenizer:

    def __init__(self):
        setup_nltk()
        self.tokenized_texts = []
        self.stop_words = stopwords.words('spanish')

    def tokenizer_text(self, texts):
        for text in texts:
            words = word_tokenize(text)
            filtered_words = self.filter_words(words)
            token_texts = self.tokenized_texts.append(filtered_words)
            print("************TEXTOS SIN FILTRAR*************")
            print(words)
            print("************TEXTOS FILTRADOS*************")
            print(filtered_words)
        print("Tokenización completada con éxito")
        return token_texts
    
    def filter_words(self, words):
        filtered_words = []
        for word in words:
            if word.lower() not in self.stop_words:
                filtered_words.append(word)
        return filtered_words