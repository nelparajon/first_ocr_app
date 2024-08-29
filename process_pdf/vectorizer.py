import sklearn
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Vectorizer():

    def __init__(self) -> None:
        self.vectorizer = CountVectorizer()

    def vectorize_doc(self, lemmatized_tokens):
    # Verificar que lemmatized_tokens sea una lista de listas de cadenas
        if not isinstance(lemmatized_tokens, list) or not all(isinstance(sublist, list) for sublist in lemmatized_tokens):
            raise TypeError(f"Se esperaba una lista de listas de listas de cadenas, se recibió {type(lemmatized_tokens).__name__}")
    
        try:
            #almacenaremos aquí las matrices
            matrices = []
            
            # Iterar sobre cada conjunto de parrafo dentro del doc
            for document_tokens in lemmatized_tokens:
                if not all(isinstance(paragraph_tokens, list) and all(isinstance(token, str) for token in paragraph_tokens) for paragraph_tokens in document_tokens):
                    raise ValueError("Cada documento debe ser una lista de listas de cadenas de texto (tokens).")
                
                document_tokens_str = [' '.join(paragraph) for paragraph in document_tokens]

                count_matrix = self.vectorizer.fit_transform(document_tokens_str)

                matrices.append(count_matrix)
            
        
            return matrices
    
        except TypeError as te:
            print(f"Vectorización. TypeError: {te}")
        except ValueError as ve:
            print(f"Vectorización. ValueError: {ve}")
        except Exception as e:
            print(f"Error inesperado en la vectorización: {e}")

    
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
        return cosine_sim
            

    def check_shape(self, docs):
        columns = docs[0].shape[1]
        for doc in docs:
            if doc.shape[1] != columns:
                raise ValueError ("Los documentos deberían tener el mismo número de columnas")
            
    def lemmatized_paragraph_to_text(self, paragraph):
        """
        Convierte una lista de lemas en un texto.
        """
        return ' '.join(paragraph)
    
    def cosine_similarity_paragraphs(self,doc1, doc2):
        """
        Compara los párrafos de dos documentos lematizados usando similitud coseno.
        
        :param doc1: Lista de listas de lemas del primer documento.
        :param doc2: Lista de listas de lemas del segundo documento.
        :return: Lista de similitudes coseno entre párrafos correspondientes de los dos documentos.
        """
        
        # Convertir los párrafos lematizados en texto.
        text1 = [self.lemmatized_paragraph_to_text(paragraph) for paragraph in doc1]
        text2 = [self.lemmatized_paragraph_to_text(paragraph) for paragraph in doc2]
        
        # Usar CountVectorizer para convertir los textos a vectores de términos.
        vectorizer = CountVectorizer().fit(text1 + text2)
        vectors1 = vectorizer.transform(text1)
        vectors2 = vectorizer.transform(text2)
        
        # Calcular la similitud coseno entre los párrafos correspondientes.
        similarities = []
        for v1, v2 in zip(vectors1, vectors2):
            similarity = cosine_similarity(v1, v2)[0][0]
            similarities.append(similarity)

        for i, similarity in enumerate(similarities):
            print(f"Porcentaje de similitud párrafo {i + 1}: {similarity * 100:.2f}%")
        
        return similarities
    
    #extraemos el calculo de similitud en una funcion para iterar sobre ella dentro de un bucle con todos los parrafos
    def calculate_paragraph_similarity(self, paragraph1_vector, paragraph2_vector):
        cosine_sim = cosine_similarity(paragraph1_vector, paragraph2_vector)
        return cosine_sim[0][0]
    
    def analyze_similarity_paragraphs(self, docs):
        try:
            if not isinstance(docs, (tuple, list)) or not all(isinstance(vector, csr_matrix) for vector in docs):
                raise TypeError(f"Se esperaba una lista de matrices durante el cálculo de similitud, pero se recibió {type(docs).__name__}")
            
            num_paragraphs = len(docs) // 2
            if len(docs) % 2 != 0:
                raise ValueError("El número de matrices en 'docs' debe ser par, ya que cada párrafo del documento 1 debe compararse con el correspondiente del documento 2.")

            doc1_paragraphs = docs[:num_paragraphs]
            doc2_paragraphs = docs[num_paragraphs:]
            
            similarities = []

            for idx, (paragraph1, paragraph2) in enumerate(zip(doc1_paragraphs, doc2_paragraphs)):
                cosine_sim = self.calculate_paragraph_similarity(paragraph1, paragraph2)
                similarities.append(cosine_sim)
                print(f"COSINE SIMILARITY between paragraph {idx} of doc1 and doc2: {cosine_sim}")

            return similarities

        except ValueError as ve:
            print(f"ValueError: {ve}")
        except TypeError as te:
            print(f"TypeError: {te}")
        except Exception as e:
            print(f"Error inesperado al comprobar la similitud: {e}")
                

    
    
    

