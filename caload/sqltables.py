import pickle
from datetime import date, datetime
from typing import List

from sqlalchemy import Index, ForeignKey, String
from sqlalchemy.dialects.mysql import MEDIUMBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ['SQLBase', 'EntityTable', 'AttributeBlobTable',
           'AnimalTable', 'RecordingTable', 'RoiTable', 'PhaseTable',
           'AttributeTable', 'AnimalValueTable', 'RecordingValueTable', 'RoiValueTable', 'PhaseValueTable',
           'TaskTable', 'TaskedEntityTable']


class SQLBase(DeclarativeBase):
    pass


# Entities

class EntityTable:
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)


class AnimalTable(EntityTable, SQLBase):
    __tablename__ = 'animals'

    id: Mapped[str] = mapped_column(String(500), unique=True)

    recordings: Mapped[List['RecordingTable']] = relationship('RecordingTable', back_populates='parent')
    attributes: Mapped[List['AnimalValueTable']] = relationship('AnimalValueTable', back_populates='entity')

    def __repr__(self):
        return f"<Animal(id={self.id}')>"


class RecordingTable(EntityTable, SQLBase):
    __tablename__ = 'recordings'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'))

    id: Mapped[str] = mapped_column(String(500))
    date: Mapped[date]

    parent: Mapped['AnimalTable'] = relationship('AnimalTable', back_populates='recordings')
    rois: Mapped[List['RoiTable']] = relationship('RoiTable', back_populates='parent')
    phases: Mapped[List['PhaseTable']] = relationship('PhaseTable', back_populates='parent')
    attributes: Mapped[List['RecordingValueTable']] = relationship('RecordingValueTable', back_populates='entity')

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
    attributes: Mapped[List['RoiValueTable']] = relationship('RoiValueTable', back_populates='entity')

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
    attributes: Mapped[List['PhaseValueTable']] = relationship('PhaseValueTable', back_populates='entity')

    __table_args__ = (
        Index('ix_unique_phase_id_per_recording', 'parent_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<Phase(id={self.id}, recording={self.parent})>"


# Attributes

class AttributeBlobTable(SQLBase):
    __tablename__ = 'attribute_blobs'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    value: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=True)


class AttributeTable:
    entity_pk: Mapped[int]
    entity: Mapped

    name: Mapped[str] = mapped_column(String(500), primary_key=True, nullable=False, index=True)

    value_blob_pk: Mapped[int]
    value_blob: Mapped['AttributeBlobTable']

    value_str: Mapped[str] = mapped_column(String(500), nullable=True)
    value_int: Mapped[int] = mapped_column(nullable=True)
    value_float: Mapped[float] = mapped_column(nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime] = mapped_column(nullable=True)
    value_path: Mapped[str] = mapped_column(String(500), nullable=True)
    column_str: Mapped[str] = mapped_column(String(500), nullable=True)

    is_persistent: Mapped[bool] = mapped_column(nullable=True)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.entity}, {self.name}, {self.value})>"

    @property
    def value(self):

        if self.column_str is None:
            return None

        # If blob, load from associated row in AttributeBlobTable
        if self.column_str == 'value_blob':
            return pickle.loads(self.value_blob.value)

        # Otherwise load from this row based on column_str
        return getattr(self, self.column_str)

    @value.setter
    def value(self, value):

        # If blob, dump to associated row in AttributeBlobTable
        if self.column_str == 'value_blob':
            self.value_blob.value = pickle.dumps(value)
            return

        # Otherwise write directly
        setattr(self, self.column_str, value)


class AnimalValueTable(AttributeTable, SQLBase):
    __tablename__ = 'animal_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'), primary_key=True, nullable=False)
    entity: Mapped['AnimalTable'] = relationship('AnimalTable', back_populates='attributes')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class RecordingValueTable(AttributeTable, SQLBase):
    __tablename__ = 'recording_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'), primary_key=True, nullable=False)
    entity: Mapped['RecordingTable'] = relationship('RecordingTable', back_populates='attributes')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class PhaseValueTable(AttributeTable, SQLBase):
    __tablename__ = 'phase_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('phases.pk'), primary_key=True, nullable=False)
    entity: Mapped['PhaseTable'] = relationship('PhaseTable', back_populates='attributes')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class RoiValueTable(AttributeTable, SQLBase):
    __tablename__ = 'roi_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('rois.pk'), primary_key=True, nullable=False)
    entity: Mapped['RoiTable'] = relationship('RoiTable', back_populates='attributes')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class TaskTable(SQLBase):
    __tablename__ = 'tasks'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    target_fun: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=True)
    target_args: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=True)

    status: Mapped[int] = mapped_column(nullable=False, default=0)  # 0: pending, 1: finished


class TaskedEntityTable(SQLBase):
    __tablename__ = 'tasked_entities'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_pk: Mapped[int] = mapped_column(ForeignKey('tasks.pk'))

    animal_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'), nullable=True)
    recording_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'), nullable=True)
    roi_pk: Mapped[int] = mapped_column(ForeignKey('rois.pk'), nullable=True)
    phase_pk: Mapped[int] = mapped_column(ForeignKey('phases.pk'), nullable=True)

    status: Mapped[int] = mapped_column(nullable=False, default=0, index=True)  # 0: pending, 1: acquired, 2: finished

    @property
    def entity_pk(self):
        return self.animal_pk or self.recording_pk or self.roi_pk or self.phase_pk


if __name__ == '__main__':
    pass