from cliente.encoder import Encoder
#script para sacar el archivo en b64

if __name__ == "__main__":
    file_path = r"C:\Users\Nel\Documents\texto_1.pdf"  
    encoded_string = Encoder.encode_file_b64(file_path)
    
    
    with open("encoded_pdf.txt", "w") as text_file:
        text_file.write(encoded_string)
    
    print("Encoded PDF saved to encoded_pdf.txt")