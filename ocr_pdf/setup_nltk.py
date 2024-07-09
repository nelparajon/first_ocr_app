import nltk
from nltk.corpus import stopwords

#configuracion y descarga de recursos que ofrece nltk
def setup_nltk():
    nltk.download("punkt") #tokenizador
    nltk.download("stopwords") #lista que contiene palabras con poco significado semantico como preposiciones, articulos, etc 

def printing_stopwords():
    print(stopwords.words('spanish'))

printing_stopwords()

