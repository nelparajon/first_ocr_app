import sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Vectorizer():

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer()

    def vectorize_doc(self, lemmatized_tokens):
        
            if not isinstance(lemmatized_tokens, list) and not all(isinstance(lem, str) for lem in lemmatized_tokens):
                raise TypeError(f"Se esperaba una lista de cadenas, pero se recibió {type(lemmatized_tokens).__name__}")
            try:
            
                #pasamos a una lista los tokens para que estén en el formato que acepta count vectorizer
                lemmatized_tokens_str = [' '.join(lem) for lem in lemmatized_tokens]
                print("LEMATIZED TOKENS \n", lemmatized_tokens_str)  

                if not isinstance(lemmatized_tokens, list) and not all(isinstance(lem, str) for lem in lemmatized_tokens_str):
                    raise ValueError(f"Se esperaba una lista de cadena de tokens, se recibió {type(lemmatized_tokens_str).__name__}") 
                #print(lemmatized_tokens_str)

                count_matrix = self.vectorizer.fit_transform(lemmatized_tokens_str)
                print("COUNT MATRIX: \n", count_matrix.toarray())
                return count_matrix
            
            except TypeError as te:
                return print(f" Vectorización. TypeError: {te}")
            except ValueError as ve:
                return print(f"Vectorización. ValueError. : {ve}")
            except Exception as e:
                return print(f"Error inesperado en la vectorización: {e}")
    
    #comprobacion de la similitud entre textos comparando los cosenos de los vectores
    def similarity_docs(self, docs):
        try:
            if not isinstance(docs, tuple) and not all(isinstance(vector, csr_matrix) for vector in docs):
                raise TypeError(f"Se esperaba una lista de matrices durante el calculo de similitud, pero se recibió {type(docs).__name__}")
            
            self.check_shape(docs)
            cosine_sim = cosine_similarity(docs)
            print("COSINE SIMILARITY: \n", cosine_sim)
        except ValueError as ve:
            return print(f"ValueError: {ve}")
        except TypeError as te:
            return print(f"TypeError: {te}")
        except Exception as e:
            return print(f"Error inesperado al comprobar la similitud: {e}")
            

    def check_shape(self, docs):
        columns = docs[0].shape[1]
        for doc in docs:
            if doc.shape[1] != columns:
                raise ValueError ("Los documentos deberían tener el mismo número de columnas")
        

    
    
    

