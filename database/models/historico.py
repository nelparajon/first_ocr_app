
from database.db import db
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

class Historico(db.Model):
    __tablename__ = 'historico'

    id: Mapped[int] = mapped_column(primary_key=True)
    fecha: Mapped[datetime] = mapped_column(nullable=False)
    estado: Mapped[str] = mapped_column(db.String(20), nullable=False)
    similitud: Mapped[str] = mapped_column(db.String(40))

    def __str__(self):
        return (
            f"Historico: id={self.id} | fecha={self.fecha} | Estado={self.estado} | Similitud={self.similitud}"
        )
    
    def __repr__(self) -> str:
        return self.__str__
    