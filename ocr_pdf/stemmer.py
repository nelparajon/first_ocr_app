import nltk
from nltk.stem import SnowballStemmer

class Stemmer:

    def __init__(self, language="spanish"):
        self.stemmer = SnowballStemmer(language)
        self.stem_tokens = []

    def stemming_tokens(self, tokenized_texts):
        stem_tokens = self.stem_tokens = []
        for tokens in tokenized_texts:
            stem = self.stemmer.stem(tokens)
            stem_tokens.append(stem)
        print(stem_tokens)
        print("Estematización completada con éxito")
        return stem_tokens
            
