from datetime import date, datetime
from typing import Any, List, Union

from sqlalchemy import Index, ForeignKey, event, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# Set WAL
@event.listens_for(Engine, 'connect')
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    # print('Set timeout and WAL')
    cursor.execute('PRAGMA journal_mode=WAL;')
    cursor.execute('PRAGMA busy_timeout=30000;')
    cursor.close()


class SQLBase(DeclarativeBase):
    pass


# Define the Entity table for animals
class Animal(SQLBase):
    __tablename__ = 'animals'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    id: Mapped[str] = mapped_column(unique=True)

    recordings: Mapped[List['Recording']] = relationship('Recording', back_populates='animal')
    attributes: Mapped[List['AnimalAttribute']] = relationship('AnimalAttribute', back_populates='entity')

    def __repr__(self):
        return f"<Animal(id={self.id}')>"


# Define the Entity table for recordings
class Recording(SQLBase):
    __tablename__ = 'recordings'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    animal_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'))

    id: Mapped[str]
    date: Mapped[date]

    animal: Mapped['Animal'] = relationship('Animal', back_populates='recordings')
    rois: Mapped[List['Roi']] = relationship('Roi', back_populates='recording')
    phases: Mapped[List['Phase']] = relationship('Phase', back_populates='recording')
    attributes: Mapped[List['RecordingAttribute']] = relationship('RecordingAttribute', back_populates='entity')

    # Define partial unique index
    __table_args__ = (
        Index('ix_unique_rec_id_per_animal_date', 'animal_pk', 'date', 'id', unique=True),
    )

    def __repr__(self):
        return f"<Recording(id={self.id}, animal={self.animal}, date={self.date})>"


class Roi(SQLBase):
    __tablename__ = 'rois'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    recording_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'))

    id: Mapped[int]

    recording: Mapped['Recording'] = relationship('Recording', back_populates='rois')
    attributes: Mapped[List['RoiAttribute']] = relationship('RoiAttribute', back_populates='entity')

    __table_args__ = (
        Index('ix_unique_roi_id_per_recording', 'recording_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<Roi(id={self.id}, recording={self.recording})>"


class Phase(SQLBase):
    __tablename__ = 'phases'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    recording_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'))

    id: Mapped[int]

    recording: Mapped['Recording'] = relationship('Recording', back_populates='phases')
    attributes: Mapped[List['PhaseAttribute']] = relationship('PhaseAttribute', back_populates='entity')

    __table_args__ = (
        Index('ix_unique_phase_id_per_recording', 'recording_pk', 'id', unique=True),
    )


# Attribute tables
class Attribute:
    name: Mapped[str] = mapped_column(primary_key=True)

    value_str: Mapped[str] = mapped_column(nullable=True)
    value_int: Mapped[int] = mapped_column(nullable=True)
    value_float: Mapped[float] = mapped_column(nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime] = mapped_column(nullable=True)
    value_path: Mapped[str] = mapped_column(nullable=True)
    value_column: Mapped[str] = mapped_column(nullable=True)

    def __repr__(self):
        return f"<{self.__class__.name}({self.entity}, {self.name}, {self.value})>"

    @property
    def value(self):
        return getattr(self, self.value_column)

    @value.setter
    def value(self, value):
        setattr(self, self.value_column, value)


class AnimalAttribute(Attribute, SQLBase):
    __tablename__ = 'animal_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'), primary_key=True)
    entity: Mapped['Animal'] = relationship('Animal', back_populates='attributes')

    # def __repr__(self):
    #     return f"<AnimalAttribute(animal={self.entity}, attribute={self.name}, value={self.value})>"


class RecordingAttribute(Attribute, SQLBase):
    __tablename__ = 'recording_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'), primary_key=True)
    entity: Mapped['Recording'] = relationship('Recording', back_populates='attributes')


class PhaseAttribute(Attribute, SQLBase):
    __tablename__ = 'phase_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('phases.pk'), primary_key=True)
    entity: Mapped['Phase'] = relationship('Phase', back_populates='attributes')

    # def __repr__(self):
    #     return f"<PhaseAttribute(phase={self.entity}, attribute={self.name}, value={self.value})>"


class RoiAttribute(Attribute, SQLBase):
    __tablename__ = 'roi_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('rois.pk'), primary_key=True)
    entity: Mapped['Roi'] = relationship('Roi', back_populates='attributes')

    # def __repr__(self):
    #     return f"<RoiAttribute(roi={self.entity}, attribute={self.name}, value={self.value})>"


if __name__ == '__main__':
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    # Create engine
    engine = create_engine('sqlite:///../test.db')
    SQLBase.metadata.create_all(engine)

    # Create a session
    session = Session(engine)

    session.close()
