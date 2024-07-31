from flask import jsonify
from database.db_manager import DbManager


class ErrorHandler:
    
    @staticmethod
    def register(app):
        @app.errorhandler(405)
        def method_not_allowed(error):
            DbManager.save_request(405, "Error: método no soportado")
            return jsonify(error="Error: método no soportado"), 405

        @app.errorhandler(400)
        def bad_request(error):
            DbManager.save_request(400, "Error con el archivo. No se han encontrado los dos documentos para comparar")
            return jsonify(error="Error con el archivo. No se han encontrado los dos documentos para comparar"), 400

        @app.errorhandler(500)
        def internal_server_error(error):
            DbManager.save_request(500, f"Error interno del servidor: {str(error)}")
            return jsonify(error=f"Error interno del servidor: {str(error)}"), 500