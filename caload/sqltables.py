import pickle
from datetime import date, datetime
from typing import List

from sqlalchemy import Index, ForeignKey, String, create_engine
from sqlalchemy.dialects.mysql import LONGBLOB, MEDIUMBLOB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = ['SQLBase', 'EntityTypeTable', 'EntityTable',
           'AttributeTable', 'TaskTable', 'EntityTaskTable']


class SQLBase(DeclarativeBase):
    pass


# Entities

class EntityTypeTable(SQLBase):
    __tablename__ = 'entity_types'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    parent_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'), nullable=True)
    parent: Mapped['EntityTypeTable'] = relationship('EntityTypeTable', back_populates='children', remote_side=[pk])
    children: Mapped[List['EntityTypeTable']] = relationship('EntityTypeTable', back_populates='parent', remote_side=[parent_pk])

    name: Mapped[str] = mapped_column(String(500), unique=True)

    entities: Mapped[List['EntityTable']] = relationship('EntityTable', back_populates='entity_type')


class EntityTable(SQLBase):
    __tablename__ = 'entities'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    entity_type_pk: Mapped[int] = mapped_column(ForeignKey('entity_types.pk'))
    parent_pk: Mapped[int] = mapped_column(ForeignKey('entities.pk'), nullable=True)

    id: Mapped[str] = mapped_column(String(500))

    # Many-to-One
    entity_type: Mapped['EntityTypeTable'] = relationship('EntityTypeTable', back_populates='entities')
    parent: Mapped['EntityTable'] = relationship('EntityTable', back_populates='children', remote_side=[pk])

    # One-to-Many
    children: Mapped[List['EntityTable']] = relationship('EntityTable', back_populates='parent', remote_side=[parent_pk])
    attributes: Mapped[List['AttributeTable']] = relationship('AttributeTable', back_populates='entity')

    __table_args__ = (
        Index('ix_unique_id_per_parent_pk', 'parent_pk', 'id', unique=True),
    )

    def __repr__(self):
        return f"<{self.entity_type.name}(id={self.id}, parent={self.parent})>"


# Attributes

class AttributeTable(SQLBase):
    __tablename__ = 'attributes'

    entity_pk: Mapped[int] = mapped_column(ForeignKey('entities.pk'), primary_key=True)
    entity: Mapped['EntityTable'] = relationship('EntityTable', back_populates='attributes')

    name: Mapped[str] = mapped_column(String(500), primary_key=True, index=True)

    value_str: Mapped[str] = mapped_column(String(500), nullable=True)
    value_int: Mapped[int] = mapped_column(nullable=True)
    value_float: Mapped[float] = mapped_column(nullable=True)
    value_bool: Mapped[bool] = mapped_column(nullable=True)
    value_date: Mapped[date] = mapped_column(nullable=True)
    value_datetime: Mapped[datetime] = mapped_column(nullable=True)
    value_blob: Mapped[bytes] = mapped_column(LONGBLOB, nullable=True)
    value_path: Mapped[str] = mapped_column(String(500), nullable=True)
    data_type: Mapped[str] = mapped_column(String(500), nullable=True)

    is_persistent: Mapped[bool] = mapped_column(nullable=True)

    def __repr__(self):
        return f"<Attribute({self.name}, {self.value}, {self.entity})>"

    @property
    def value(self):

        if self.data_type is None:
            return None

        # If blob, load from associated row in AttributeBlobTable
        if self.data_type == 'blob':
            return pickle.loads(self.value_blob)

        # Otherwise load from this row based on column_str
        return getattr(self, f'value_{self.data_type}')

    @value.setter
    def value(self, value):

        # If blob, dump to associated row in AttributeBlobTable
        if self.data_type == 'blob':
            self.value_blob = pickle.dumps(value)
            return

        # Otherwise write directly
        setattr(self, f'value_{self.data_type}', value)


# Tasks

class TaskTable(SQLBase):
    __tablename__ = 'tasks'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    target_fun: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=True)
    target_args: Mapped[bytes] = mapped_column(MEDIUMBLOB, nullable=True)

    status: Mapped[int] = mapped_column(nullable=False, default=0)  # 0: pending, 1: finished


class EntityTaskTable(SQLBase):
    __tablename__ = 'entity_tasks'

    pk: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_pk: Mapped[int] = mapped_column(ForeignKey('tasks.pk'))

    entity_pk: Mapped[int] = mapped_column(ForeignKey('entities.pk'), nullable=False)

    status: Mapped[int] = mapped_column(nullable=False, default=0, index=True)  # 0: pending, 1: acquired, 2: finished

