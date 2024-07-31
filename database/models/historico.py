
from database.db import db
from sqlalchemy.orm import Mapped, mapped_column
import time
from datetime import datetime

class Historico(db.Model):
    __tablename__ = 'historico'

    id: Mapped[int] = mapped_column(primary_key=True)
    fecha: Mapped[datetime] = mapped_column(nullable=False, default=lambda: Historico.get_local_time())
    estado: Mapped[str] = mapped_column(db.String(50), nullable=False)
    mensaje: Mapped[str] = mapped_column(db.String(200))
    porcentaje_similitud: Mapped[float] = mapped_column(db.Numeric(5,2))

    @staticmethod
    def get_local_time():
        local_time = time.localtime()
        return datetime.fromtimestamp(time.mktime(local_time))

    def __init__(self, estado, mensaje, porcentaje_similitud):
        self.fecha = self.get_local_time()  # Establece la fecha actual por defecto
        self.estado = estado
        self.porcentaje_similitud = porcentaje_similitud
        self.mensaje = mensaje

    def __str__(self):
        return (
            f"Historico: id={self.id} | fecha={self.fecha} | Estado={self.estado}"
        )

    def __repr__(self) -> str:
        return self.__str__()
    
    
    