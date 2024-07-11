import nltk
from nltk.corpus import stopwords

#configuracion y descarga de recursos que ofrece nltk
def setup_nltk():
    #nltk.download("punkt") #tokenizador oraciones
    nltk.download("stopwords") #lista que contiene palabras con poco significado semantico como preposiciones, articulos, etc 
    nltk.download("wordnet")

def printing_stopwords():
    sw = stopwords.words('spanish')
    print(sw)
    
   

printing_stopwords()

