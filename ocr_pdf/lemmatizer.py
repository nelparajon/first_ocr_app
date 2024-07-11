import nltk
from setup_nltk import setup_nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Lemmatizer:

    def __init__(self) -> None:
        setup_nltk()
        self.lemmatizer = WordNetLemmatizer()
        tag_dict = {

        }

    def lemmatizing_words(self, tokens):
        lemmatized_words = []
        for words in tokens:
            for word in words:
                lem = self.lemmatizer.lemmatize(word)
                lemmatized_words.append(lem)
        print("*********************LEMMATIZING***************")
        print(lemmatized_words)
        return lemmatized_words