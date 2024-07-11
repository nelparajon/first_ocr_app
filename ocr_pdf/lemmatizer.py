import nltk
from nltk import pos_tag
from setup_nltk import setup_nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Lemmatizer:
    def __init__(self):
        setup_nltk()  # Configura NLTK descargando los recursos necesarios
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        """Map Treebank POS tag to WordNet POS tag"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def lemmatizing_words(self, tokenized_texts):
        lemmatized_texts = []  # Almacenamos los tokens en una lista
        for tokens in tokenized_texts:
            pos_tagged_tokens = pos_tag(tokens)  # Etiquetado POS
            lemmatized_words = []
            for word, tag in pos_tagged_tokens:
                wordnet_pos = self.get_wordnet_pos(tag)
                lem = self.lemmatizer.lemmatize(word, wordnet_pos)
                lemmatized_words.append(lem)
            lemmatized_texts.append(lemmatized_words)
        print("*********************LEMMATIZING***************")
        print(lemmatized_texts)
        return lemmatized_texts
    




