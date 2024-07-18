import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Vectorizer():

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer()

    def vectorize_doc(self, lemmatized_tokens):
        #pasamos a una lista los tokens para que estén en el formato que acepta count vectorizer
        lemmatized_tokens_str = [' '.join(lem) for lem in lemmatized_tokens] 
        print(lemmatized_tokens_str)
        count_matrix = self.vectorizer.fit_transform(lemmatized_tokens_str)
        print("COUNT MATRIX: \n", count_matrix.toarray())
        return count_matrix.toarray()
    
    #comprobacion de la similitud entre textos comparando los cosenos de los vectores
    def similarity_docs(self, docs):
        cosine_sim = cosine_similarity(docs)
        print("COSINE SIMILARITY: \n", cosine_sim)

    
    
    

