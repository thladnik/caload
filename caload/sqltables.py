import pickle
from datetime import date, datetime
from typing import List

from sqlalchemy import Index, ForeignKey, String
from sqlalchemy.dialects.mysql import MEDIUMBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ['SQLBase', 'EntityTable', 'AttributeBlobTable',
           'AttributeTable', 'AnimalTable', 'RecordingTable', 'RoiTable', 'PhaseTable',
           'AttributeValueTable', 'AnimalValueTable', 'RecordingValueTable', 'RoiValueTable', 'PhaseValueTable']


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

class AttributeTable(SQLBase):
    __tablename__ = 'attributes'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    name: Mapped[str] = mapped_column(String(500), unique=True)

    animal_values: Mapped[List['AnimalValueTable']] = relationship('AnimalValueTable', back_populates='attribute')
    recording_values: Mapped[List['RecordingValueTable']] = relationship('RecordingValueTable', back_populates='attribute')
    phase_values: Mapped[List['PhaseValueTable']] = relationship('PhaseValueTable', back_populates='attribute')
    roi_values: Mapped[List['RoiValueTable']] = relationship('RoiValueTable', back_populates='attribute')


class AttributeBlobTable(SQLBase):
    __tablename__ = 'attribute_blobs'
    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    value: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=True)


class AttributeValueTable:
    entity_pk: Mapped[int]
    entity: Mapped

    attribute_pk: Mapped[int]
    attribute: Mapped

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
        return f"<{self.__class__.__name__}({self.entity}, {self.attribute.name}, {self.value})>"

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


class AnimalValueTable(AttributeValueTable, SQLBase):
    __tablename__ = 'animal_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('animals.pk'), primary_key=True)
    entity: Mapped['AnimalTable'] = relationship('AnimalTable', back_populates='attributes')

    attribute_pk: Mapped[int] = mapped_column(ForeignKey('attributes.pk'), primary_key=True)
    attribute: Mapped['AttributeTable'] = relationship('AttributeTable', back_populates='animal_values')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class RecordingValueTable(AttributeValueTable, SQLBase):
    __tablename__ = 'recording_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('recordings.pk'), primary_key=True)
    entity: Mapped['RecordingTable'] = relationship('RecordingTable', back_populates='attributes')

    attribute_pk: Mapped[int] = mapped_column(ForeignKey('attributes.pk'), primary_key=True)
    attribute: Mapped['AttributeTable'] = relationship('AttributeTable', back_populates='recording_values')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class PhaseValueTable(AttributeValueTable, SQLBase):
    __tablename__ = 'phase_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('phases.pk'), primary_key=True)
    entity: Mapped['PhaseTable'] = relationship('PhaseTable', back_populates='attributes')

    attribute_pk: Mapped[int] = mapped_column(ForeignKey('attributes.pk'), primary_key=True)
    attribute: Mapped['AttributeTable'] = relationship('AttributeTable', back_populates='phase_values')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


class RoiValueTable(AttributeValueTable, SQLBase):
    __tablename__ = 'roi_attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('rois.pk'), primary_key=True)
    entity: Mapped['RoiTable'] = relationship('RoiTable', back_populates='attributes')

    attribute_pk: Mapped[int] = mapped_column(ForeignKey('attributes.pk'), primary_key=True)
    attribute: Mapped['AttributeTable'] = relationship('AttributeTable', back_populates='roi_values')

    value_blob_pk: Mapped[int] = mapped_column(ForeignKey('attribute_blobs.pk'), nullable=True)
    value_blob: Mapped['AttributeBlobTable'] = relationship('AttributeBlobTable')


if __name__ == '__main__':
    pass