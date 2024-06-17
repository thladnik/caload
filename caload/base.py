from __future__ import annotations

import os.path
from abc import  abstractmethod
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, Union

import h5py
import numpy as np
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import Session

from caload import sqltables as sql
from caload import utils

__all__ = ['Analysis', 'Entity', 'Animal', 'Recording', 'Roi', 'Mode']


class Mode(Enum):
    create = 1
    analyse = 2


class Analysis:
    mode: Mode
    analysis_path: str
    sql_engine: Engine
    session: Session
    index: h5py.File
    cache_attr_names_to_pk: Dict[str, int] = {}
    entities: Dict[
        Union[sql.Animal, sql.Recording, sql.Roi, sql.Phase], Entity] = {}

    def __init__(self, path: str, mode: Mode = Mode.analyse):
        self.mode = mode
        self.analysis_path = Path(path).as_posix()

        print(f'Open analysis {self.analysis_path}')

        # Create engine
        engine = create_engine(f'sqlite:///{self.analysis_path}/metadata.db', echo=False)

        sql.SQLBase.metadata.create_all(engine)

        # Create a session
        self.session = Session(engine)

    def add_animal(self, animal_id: str) -> Animal:
        return Animal.create(analysis=self, animal_id=animal_id)

    def animal(self, *args, **kwargs) -> List[Animal]:
        return Animal.filter(self, *args, **kwargs)

    def recordings(self, *args, **kwargs) -> List[Recording]:
        return Recording.filter(self, *args, **kwargs)

    def rois(self, *args, **kwargs) -> List[Roi]:
        return Roi.filter(self, *args, **kwargs)

    def add_entity(self, entity: Entity):
        entity_path = f'{self.analysis_path}/{entity.path}'

        # Create directoty of necessary
        if not os.path.exists(entity_path):
            os.makedirs(entity_path)

        # Create data file
        path = f'{entity_path}/data.hdf5'
        with h5py.File(path, 'w') as _:
            pass

    def get_entity_attribute(self, entity: Entity, name: str):
        with h5py.File(f'{self.analysis_path}/{entity.path}/data.hdf5', 'r') as f:
            return f[name][:]

    def set_entity_attribute(self, entity: Entity, name: str, value: Any):
        path = f'{self.analysis_path}/{entity.path}/data.hdf5'
        # TODO: handle value casts, e.g. serialize lists, tuples and other objects
        with h5py.File(path, 'a') as f:
            if name not in f:
                f.create_dataset(name, data=value)
            else:
                f[name][:] = value
        # self.index[f'{entity.path}/link'].create_dataset(name, data=value)


class Entity:
    _analysis: Analysis
    _attribute_table = Union[Type[sql.Animal], Type[sql.Recording], Type[sql.Roi]]

    def __init__(self, row: Union[sql.Animal, sql.Recording, sql.Roi], analysis: Analysis):
        self._analysis = analysis
        self._row: Union[sql.Animal, sql.Recording, sql.Roi] = row
        self._scalar_attributes: Dict[str, Union[sql.AnimalAttribute, sql.RecordingAttribute, sql.RoiAttribute]] = {}

        # In create mode there are no entity attributes yet
        if not self.analysis.mode == Mode.create:
            self._update_scalar_attributes()

    def __new__(cls, row: sql.Animal, analysis: Analysis):
        if not analysis.mode == Mode.create:
            # Create new instance if none exists
            if row not in analysis.entities:
                instance = super(cls.__class__, cls).__new__(cls)
                analysis.entities[row] = instance

            # Return instance
            return analysis.entities[row]
        return super(cls).__new__(cls)

    def _update_scalar_attributes(self):
        query = (self.analysis.session.query(self._attribute_table)
                 .filter(self._attribute_table.entity_pk == self.row.pk))

        for row in query.all():
            self._scalar_attributes[row.name] = row.value

    def __getitem__(self, item):

        if item in self._scalar_attributes:
            return self._scalar_attributes[item]

        return self.analysis.get_entity_attribute(self, item)

    def __setitem__(self, key: str, value: Any):

        # Find corresponding builtin python scalar type for numpy scalars and shape () arrays
        if isinstance(value, np.generic) or (isinstance(value, np.ndarray) and np.squeeze(value).shape == ()):
            # Convert to the corresponding Python built-in type using the item() method
            value = value.item()

        # Set scalars
        if type(value) in (str, float, int, bool, date, datetime):

            # Set local cache
            self._scalar_attributes[key] = value

            # Query attribute row
            row = None
            if not self.analysis.mode == Mode.create:
                row = self._query_attribute_row(key)

            # Create new row
            if row is None:
                value_type_map = {str: 'str', float: 'float', int: 'int',
                                  bool: 'bool', date: 'date', datetime: 'datetime'}
                value_type_str = value_type_map.get(type(value))
                row = self._attribute_table(entity_pk=self.row.pk, name=key, value_column=f'value_{value_type_str}')
                self.analysis.session.add(row)

            # Set value
            row.value = value

            # Commit changes
            self.analysis.session.commit()

        # Set arrays
        elif isinstance(value, (list, tuple, np.ndarray)):
            self.analysis.set_entity_attribute(self, key, value)
        else:
            print(f'No mapping for attribute {key} of type {type(value)}')

    def update(self, data: Dict[str, Any]):
        """Implement update method for usage like in dict.update"""
        for key, value in data.items():
            self[key] = value

    @property
    @abstractmethod
    def path(self) -> str:
        pass

    @property
    def row(self) -> sql.Animal:
        return self._row

    @property
    def analysis(self) -> Analysis:
        return self._analysis

    def _query_attribute_row(self, key: str) -> Union[sql.AnimalAttribute, sql.RecordingAttribute, sql.RoiAttribute, None]:

        # Build Entity-specific query
        query = (self.analysis.session.query(self._attribute_table)
                 .filter(self._attribute_table.entity_pk == self.row.pk)
                 .filter(self._attribute_table.name == key))

        # Return None
        if query.count() == 0:
            return

        # Return result
        return query.one()


class Animal(Entity):
    _attribute_table = sql.AnimalAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)

    def __repr__(self):
        return f"Animal(id='{self.id}')"

    @staticmethod
    def create(animal_id: str, analysis: Analysis):
        row = sql.Animal(id=animal_id)
        analysis.session.add(row)
        analysis.session.commit()
        entity = Animal(row=row, analysis=analysis)
        analysis.add_entity(entity)
        return entity

    @property
    def path(self) -> str:
        return f'animal/{self.id}'

    @property
    def id(self) -> str:
        return self._row.id

    def add_recording(self, *args, **kwargs) -> Recording:
        return Recording.create(*args, animal=self, analysis=self.analysis, **kwargs)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None) -> List[Animal]:
        result = _filter(analysis,
                         sql.Animal,
                         [],
                         sql.AnimalAttribute,
                         *attr_filters,
                         animal_id=animal_id)

        return [Animal(row=row, analysis=analysis) for row in result]


class Recording(Entity):
    _attribute_table = sql.RecordingAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self._animal: sql.Animal = self.row.animal

    def __repr__(self):
        return f"Recording(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(animal: Animal, rec_date: date, rec_id: str, analysis: Analysis):
        row = sql.Recording(animal_pk=animal.row.pk, date=utils.parse_date(rec_date), id=rec_id)
        analysis.session.add(row)
        analysis.session.commit()
        entity = Recording(row=row, analysis=analysis)
        analysis.add_entity(entity)
        return entity

    def add_roi(self, *args, **kwargs) -> Roi:
        return Roi.create(*args, recording=self, analysis=self.analysis, **kwargs)

    def add_phase(self, *args, **kwargs) -> Phase:
        return Phase.create(*args, recording=self, analysis=self.analysis, **kwargs)

    @property
    def path(self) -> str:
        return f'{self.animal.path}/recording/{self.rec_date}_{self.id}'

    @property
    def id(self) -> str:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self._animal.id

    @property
    def rec_date(self) -> date:
        return self._row.date

    @property
    def row(self) -> sql.Recording:
        return self._row

    @property
    def animal(self) -> Animal:
        return Animal(analysis=self.analysis, row=self._animal)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None) -> List[Recording]:
        result = _filter(analysis,
                         sql.Recording,
                         [sql.Animal],
                         sql.RecordingAttribute,
                         *attr_filters,
                         animal_id=animal_id,
                         rec_date=rec_date,
                         rec_id=rec_id)

        return [Recording(row=row, analysis=analysis) for row in result]


class Phase(Entity):
    _attribute_table = sql.PhaseAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.analysis.session.flush()
        self._recording: sql.Recording = self.row.recording
        self._animal: sql.Animal = self._recording.animal

    def __repr__(self):
        return f"Phase(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"rec_id='{self.rec_id}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(recording: Recording, phase_id: int, analysis: Analysis):
        row = sql.Phase(recording_pk=recording.row.pk, id=phase_id)
        analysis.session.add(row)
        entity = Phase(row=row, analysis=analysis)
        analysis.add_entity(entity)
        analysis.session.commit()
        return entity

    @property
    def path(self) -> str:
        return f'{self.recording.path}/phases/{self.id}'

    @property
    def id(self) -> int:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self._animal.id

    @property
    def rec_date(self) -> date:
        return self._recording.date

    @property
    def rec_id(self) -> str:
        return self._recording.id

    @property
    def row(self) -> sql.Roi:
        return self._row

    @property
    def animal(self) -> Animal:
        return Animal(row=self._animal)

    @property
    def recording(self) -> Recording:
        return Recording(analysis=self.analysis, row=self._recording)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None,
               phase_id: int = None, ) -> List[Phase]:
        result = _filter(analysis,
                         sql.Phase,
                         [sql.Recording, sql.Animal],
                         sql.PhaseAttribute,
                         *attr_filters,
                         animal_id=animal_id,
                         rec_date=rec_date,
                         rec_id=rec_id,
                         phase_id=phase_id)

        return [Phase(row=row, analysis=analysis) for row in result]


class Roi(Entity):
    _attribute_table = sql.RoiAttribute

    def __init__(self, *args, **kwargs):
        Entity.__init__(self, *args, **kwargs)
        self.analysis.session.flush()
        self._recording: sql.Recording = self.row.recording
        self._animal: sql.Animal = self._recording.animal

    def __repr__(self):
        return f"Roi(id={self.id}, " \
               f"rec_date='{self.rec_date}', " \
               f"rec_id='{self.rec_id}', " \
               f"animal_id='{self.animal_id}')"

    @staticmethod
    def create(recording: Recording, roi_id: int, analysis: Analysis):
        row = sql.Roi(recording_pk=recording.row.pk, id=roi_id)
        analysis.session.add(row)
        entity = Roi(row=row, analysis=analysis)
        analysis.add_entity(entity)
        analysis.session.commit()
        return entity

    @property
    def path(self) -> str:
        return f'{self.recording.path}/rois/{self.id}'

    @property
    def id(self) -> int:
        return self._row.id

    @property
    def animal_id(self) -> str:
        return self._animal.id

    @property
    def rec_date(self) -> date:
        return self._recording.date

    @property
    def rec_id(self) -> str:
        return self._recording.id

    @property
    def row(self) -> sql.Roi:
        return self._row

    @property
    def animal(self) -> Animal:
        return Animal(row=self._animal)

    @property
    def recording(self) -> Recording:
        return Recording(analysis=self.analysis, row=self._recording)

    @classmethod
    def filter(cls,
               analysis: Analysis,
               *attr_filters,
               animal_id: str = None,
               rec_date: Union[str, date, datetime] = None,
               rec_id: str = None,
               roi_id: int = None, ) -> List[Roi]:
        result = _filter(analysis,
                         sql.Roi,
                         [sql.Recording, sql.Animal],
                         sql.RoiAttribute,
                         *attr_filters,
                         animal_id=animal_id,
                         rec_date=rec_date,
                         rec_id=rec_id,
                         roi_id=roi_id)

        return [Roi(row=row, analysis=analysis) for row in result]


def _filter(analysis: Analysis,
            base_table: Type[sql.SQLBase],
            joined_tables: List[Type[sql.SQLBase]],
            attribute_table: Type[sql.SQLBase],
            *attribute_filters: Tuple[(str, str, Any)],
            animal_id: str = None,
            rec_date: Union[str, date, datetime] = None,
            rec_id: str = None,
            roi_id: int = None,
            phase_id: int = None) -> List[Union[sql.Animal, sql.Recording, sql.Roi]]:

    # Convert date
    rec_date = utils.parse_date(rec_date)

    # Join parent tables
    query = analysis.session.query(base_table)
    for t in joined_tables:
        query = query.join(t)

    # Filter
    if animal_id is not None:
        query = query.filter(sql.Animal.id == animal_id)
    if rec_date is not None:
        query = query.filter(sql.Recording.date == rec_date)
    if rec_id is not None:
        query = query.filter(sql.Recording.id == rec_id)
    if roi_id is not None:
        query = query.filter(sql.Roi.id == roi_id)
    if phase_id is not None:
        query = query.filter(sql.Phase.id == phase_id)

    # Apply attribute filters
    if len(attribute_filters) > 0:

        query = query.join(attribute_table)

        for name, comp, value in attribute_filters:
            attr_value_field = None
            if isinstance(value, float):
                attr_value_field = attribute_table.value_float
            elif isinstance(value, int):
                attr_value_field = attribute_table.value_int
            elif isinstance(value, str):
                attr_value_field = attribute_table.value_string
            elif isinstance(value, bool):
                attr_value_field = attribute_table.value_bool

            if attr_value_field is None:
                raise TypeError(f'Invalid type "{type(value)}" to filter for in attributes')

            if valid := comp == 'l':
                query = query.filter(attr_value_field < value)
            elif valid := comp == 'le':
                query = query.filter(attr_value_field <= value)
            elif valid := comp == 'e':
                query = query.filter(attr_value_field == value)
            elif valid := comp == 'ge':
                query = query.filter(attr_value_field >= value)
            elif valid := comp == 'g':
                query = query.filter(attr_value_field > value)

            # Filter for attribute name if any value filter was valid
            if valid:
                query = query.filter(attribute_table.name == name)

    # Return type should be List[sql.SQLBase], PyCharm seems to get confused here...
    return query.all()  # type: ignore


if __name__ == '__main__':
    pass
