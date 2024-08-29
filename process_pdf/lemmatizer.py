import nltk
from nltk import pos_tag
from process_pdf.setup import setup_nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

class Lemmatizer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        """Mapear Treebank POS Tag a Wordnet POS Tag"""
        try:
            if not isinstance(treebank_tag, str):
                raise TypeError(f"Se esperaba una cadena de texto, pero se recibió {type(treebank_tag).__name__}")

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
        
        except TypeError as te:
            print(f"TypeError: {te}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"Error inesperado, posible problema con POS Tag: {e}")
        
    
    def lemmatizing_words(self, tokenized_texts):
        try:    
            if not isinstance(tokenized_texts, list):
                return TypeError(f"Los tokens deben de ser una lista de cadenas, se recibió {type(tokenized_texts).__name__}")
            
            if not all(isinstance(token, str) for token in tokenized_texts):
                raise ValueError("Todos los elementos de la lista deben ser cadenas de texto.")

            
            lemmatized_tokens = []
            pos_tagged_tokens = pos_tag(tokenized_texts)
            for word, tag in pos_tagged_tokens:
                wordnet_pos = self.get_wordnet_pos(tag)
                lem = self.lemmatizer.lemmatize(word, wordnet_pos)
                lemmatized_tokens.append(lem)
            print("LEMATIZACION REALIZADA CON ÉXITO")
            
            return lemmatized_tokens
        except TypeError as te:
            return print(f"TypeError: {te}")
        except ValueError as ve:
            return print(f"ValueError: {ve}")
        except Exception as e:
            return print(f"Error inesperado en la lematización: {e}")
    
    def lem_Words(self, token_texts):
        try:
               
            if not isinstance(token_texts, list):
                return TypeError(f"Los tokens deben de ser una lista de cadenas, se recibió {type(token_texts).__name__}")
            
            if not all(isinstance(tokens, list) and all(isinstance(t, str) for t in tokens) for tokens in token_texts):
                raise ValueError("Todos los elementos de la lista deben ser listas de cadenas de texto.")
            lemmatized_tokens = []
            for tokens in token_texts:
                lems_list = [] 
                pos_tagged_tokens = pos_tag(tokens)
                for word, tag in pos_tagged_tokens:
                    wordnet_pos = self.get_wordnet_pos(tag)
                    lem = self.lemmatizer.lemmatize(word, wordnet_pos)
                    lems_list.append(lem)
                lemmatized_tokens.append(lems_list)
                print("LEMATIZACION REALIZADA CON ÉXITO")
                
            return lemmatized_tokens
        except TypeError as te:
            return print(f"TypeError: {te}")
        except ValueError as ve:
            return print(f"ValueError: {ve}")
        except Exception as e:
            return print(f"Error inesperado en la lematización: {e}")



    




