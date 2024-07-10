import os

class AnalyzeText:
    def __init__(self, output_folder='output_texts'):
        self.output_folder = output_folder

    def clean_and_split_text(self, text):
        new_text = ''.join([c if c.isalnum() else ' ' for c in text]).lower()
        return new_text.split()
    
    def count_words(self, texts):
        total_word_count = 0
        for i, text in enumerate(texts):
            words = text.split()
            word_count = len(words)
            total_word_count += word_count
            #print(f"Palabras Totales en el texto {i}: ", word_count)
        
        #print("Número total de palabras en todos los textos: ", total_word_count)
        return total_word_count, word_count
        
    def words_used(self, texts):
        #print("número de veces de cada palabra repetida: \n")
        word_count = {}
        for text in texts:
            words = self.clean_and_split_text(text)
            for word in words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        
        repeated_words = {}
        for word, count in word_count.items():
            if count >= 2:
                repeated_words[word] = count
        #for word, count in repeated_words.items():
            #print(f"{word}: {count} veces")
        return repeated_words
    
    def words_most_used(self, words, limitador=5):
        most_used_words = {}
        #print("Palabras más usadas: \n")
        for w, c in words.items():
            if c >= limitador:
                most_used_words[w] = c
                #print(f"{w}: {c}")
    
    def most_used_word_in_each_text(self, texts):
        most_used_words = {}
        for i, text in enumerate(texts):
            word_count = {}
            words = self.clean_and_split_text(text)
            for word in words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
            most_used_word = max(word_count, key=word_count.get)
            most_used_words[f"texto_{i}"] = (most_used_word, word_count[most_used_word])
        #print(most_used_words)
        return most_used_words




                
            
        


