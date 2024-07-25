from encoder import Encoder

if __name__ == '__main__':

    # Ruta al archivo que deseas enviar
    file_path = 'C:/Users/Nel/Desktop/vectorizacion.pdf'
    encoder = Encoder(file_path)

    # Codificar el archivo en base64
    encoded_file = encoder.encode_file_b64()
    print(encoded_file)