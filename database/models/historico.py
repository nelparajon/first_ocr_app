
from database.db import db
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

class Historico(db.Model):
    __tablename__ = 'historico'

    id: Mapped[int] = mapped_column(primary_key=True)
    fecha: Mapped[datetime] = mapped_column(nullable=False)
    estado: Mapped[str] = mapped_column(db.String(20), nullable=False)
    #similitud: Mapped[float] = mapped_column(db.Float)
    mensaje: Mapped[str] = mapped_column(db.String(200))

    def __init__(self, estado, mensaje):
        self.fecha = datetime.utcnow()  # Establece la fecha actual por defecto
        self.estado = estado
        #self.similitud = similitud
        self.mensaje = mensaje

    def __str__(self):
        return (
            f"Historico: id={self.id} | fecha={self.fecha} | Estado={self.estado}"
        )
    
    def __repr__(self) -> str:
        return self.__str__
    