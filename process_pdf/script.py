from encoder import Encoder

if __name__ == "__main__":
    file_path = r"C:\Users\Nel\Desktop\vectorization_2.pdf"  # Replace with the path to your PDF file
    encoded_string = Encoder.encode_file_b64(file_path)
    
    # Save the encoded string to a file
    with open("encoded_pdf.txt", "w") as text_file:
        text_file.write(encoded_string)
    
    print("Encoded PDF saved to encoded_pdf.txt")