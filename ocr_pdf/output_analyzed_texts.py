import os
class AnalyzedText:
    def __init__(self, folder_path='analized_texts') -> None:
        self.folder_path = folder_path

    def save_lemmatized_texts(self, text):
    
        os.makedirs(self.folder_path, exist_ok=True)
   
        file_path = os.path.join(self.folder_path, "lemmatize_text.txt")
    
        with open(file_path, "w") as file:
            for line in text:
                file.write(line + "\n")