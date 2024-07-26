import pickle
from datetime import date, datetime
from typing import List

from sqlalchemy import Index, ForeignKey, event, Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ['SQLBase', 'EntityTable', 'AttributeTable',
           'AnimalTable', 'RecordingTable', 'RoiTable', 'PhaseTable',
           'AnimalAttributeTable', 'RecordingAttributeTable', 'RoiAttributeTable', 'PhaseAttributeTable']


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


class EntityTable:
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)


class AnimalTable(EntityTable, SQLBase):
    __tablename__ = 'animals'

    id: Mapped[str] = mapped_column(unique=True)

    recordings: Mapped[List['RecordingTable']] = relationship('RecordingTable', back_populates='parent')
    attributes: Mapped[List['AnimalAttributeTable']] = relationship('AnimalAttributeTable', back_populates='entity')

    def __repr__(self):
        return f"<Animal(id={self.id}')>"


class RecordingTable(EntityTable, SQLBase):
    __tablename__ = 'recordings'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'))

    id: Mapped[str]
    date: Mapped[date]

    parent: Mapped['AnimalTable'] = relationship('AnimalTable', back_populates='recordings')
    rois: Mapped[List['RoiTable']] = relationship('RoiTable', back_populates='parent')
    phases: Mapped[List['PhaseTable']] = relationship('PhaseTable', back_populates='parent')
    attributes: Mapped[List['RecordingAttributeTable']] = relationship('RecordingAttributeTable', back_populates='entity')

    # Define partial unique index
    __table_args__ = (
        Index('ix_unique_rec_id_per_animal_date', 'parent_pk', 'date', 'id', unique=True),
    )

    def __repr__(self):
        return f"<Recording(id={self.id}, animal={self.parent}, date={self.date})>"


class RoiTable(EntityTable, SQLBase):
    __tablename__ = 'rois'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'))

    id: Mapped[int]

    parent: Mapped['RecordingTable'] = relationship('RecordingTable', back_populates='rois')
    attributes: Mapped[List['RoiAttributeTable']] = relationship('RoiAttributeTable', back_populates='entity')

    __table_args__ = (
        Index('ix_unique_roi_id_per_recording', 'parent_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<Roi(id={self.id}, recording={self.parent})>"


class PhaseTable(EntityTable, SQLBase):
    __tablename__ = 'phases'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'))

    id: Mapped[int]

    parent: Mapped['RecordingTable'] = relationship('RecordingTable', back_populates='phases')
    attributes: Mapped[List['PhaseAttributeTable']] = relationship('PhaseAttributeTable', back_populates='entity')

    __table_args__ = (
        Index('ix_unique_phase_id_per_recording', 'parent_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<Phase(id={self.id}, recording={self.parent})>"


class AttributeTable:
    entity_pk: Mapped[int]
    entity: Mapped

    name: Mapped[str] = mapped_column(primary_key=True)

    value_str: Mapped[str] = mapped_column(nullable=True)
    value_int: Mapped[int] = mapped_column(nullable=True)
    value_float: Mapped[float] = mapped_column(nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime] = mapped_column(nullable=True)
    value_blob: Mapped[bytes] = mapped_column(nullable=True)
    value_path: Mapped[str] = mapped_column(nullable=True)
    column_str: Mapped[str] = mapped_column(nullable=True)

    is_persistent: Mapped[bool] = mapped_column(nullable=True)

    def __repr__(self):
        return f"<{self.__class__.name}({self.entity}, {self.name}, {self.value})>"

    @property
    def value(self):
        if self.column_str is not None:
            if self.column_str == 'value_blob':
                return pickle.loads(self.value_blob)
            return getattr(self, self.column_str)
        return None

    @value.setter
    def value(self, value):
        if self.column_str == 'value_blob':
            value = pickle.dumps(value)
        setattr(self, self.column_str, value)


class AnimalAttributeTable(AttributeTable, SQLBase):
    __tablename__ = 'animal_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'), primary_key=True)
    entity: Mapped['AnimalTable'] = relationship('AnimalTable', back_populates='attributes')


class RecordingAttributeTable(AttributeTable, SQLBase):
    __tablename__ = 'recording_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'), primary_key=True)
    entity: Mapped['RecordingTable'] = relationship('RecordingTable', back_populates='attributes')


class PhaseAttributeTable(AttributeTable, SQLBase):
    __tablename__ = 'phase_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('phases.pk'), primary_key=True)
    entity: Mapped['PhaseTable'] = relationship('PhaseTable', back_populates='attributes')


class RoiAttributeTable(AttributeTable, SQLBase):
    __tablename__ = 'roi_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('rois.pk'), primary_key=True)
    entity: Mapped['RoiTable'] = relationship('RoiTable', back_populates='attributes')


if __name__ == '__main__':
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    # Create engine
    engine = create_engine('sqlite:///test.db')
    SQLBase.metadata.create_all(engine)

    # Create a session
    session = Session(engine)

    session.close()
