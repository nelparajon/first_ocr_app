import sys
import os

#Agregar el directorio principal al PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'process_pdf')))

from cliente.app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
