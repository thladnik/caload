from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Dict, Union

import h5py
import yaml
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from caload.entities import Animal, EntityCollection, Recording, Roi, Phase
from caload.sqltables import SQLBase, AnimalTable, RecordingTable, RoiTable, PhaseTable

__all__ = ['Analysis', 'Mode', 'open_analysis']


class Mode(Enum):
    create = 1
    analyse = 2


class Analysis:
    _default_bulk_format = 'hdf5'
    mode: Mode
    analysis_path: str
    sql_engine: Engine
    session: Session
    index: h5py.File
    write_timeout = 3.  # s
    lazy_init: bool
    echo: bool
    max_blob_size: int
    entities: Dict[tuple, Union[Animal, Recording, Roi, Phase]]

    def __init__(self, path: str, mode: Mode = Mode.analyse, bulk_format: str = None,
                 lazy_init: bool = False, echo: bool = False, max_blob_size: int = 2 ** 22):

        # Set mode
        self.mode = mode
        self.is_create_mode = mode is Mode.create

        # Set path as posix
        self._analysis_path = Path(path).as_posix()

        # Set up entity dictionary
        self.entities = {}

        if self.is_create_mode:
            if bulk_format is None:
                bulk_format = self._default_bulk_format

            with open(f'{self.analysis_path}/configuration.yaml', 'w') as f:
                self.config = {'bulk_format': bulk_format}
                yaml.safe_dump(self.config, f)

        else:
            if bulk_format is not None:
                print('WARNING: Bulk format may only be set during creation')

            with open(f'{self.analysis_path}/configuration.yaml', 'r') as f:
                self.config = yaml.safe_load(f)

        # Set optionals
        self.lazy_init = lazy_init
        self.echo = echo
        self.max_blob_size = max_blob_size

        # Open SQL session by default
        if not lazy_init:
            self.open_session()
        # else:
        #     if self.is_create_mode:
        #         raise Exception('Cannot lazily initialize in create mode')
        #
        # # Preload all entities for fast lookup of existing entities during creation
        # if self.is_create_mode:
        #     for row in self.session.query(AnimalTable).all():
        #         Animal(row=row, analysis=self)
        #
        #     for row in self.session.query(RecordingTable).all():
        #         Recording(row=row, analysis=self)
        #
        #     for row in self.session.query(RoiTable).all():
        #         Roi(row=row, analysis=self)
        #
        #     for row in self.session.query(PhaseTable).all():
        #         Phase(row=row, analysis=self)

    def __repr__(self):
        return f"Analysis('{self.analysis_path}')"

    @property
    def analysis_path(self):
        return self._analysis_path

    @property
    def bulk_format(self):
        return self.config['bulk_format']

    def open_session(self):

        # Create engine
        engine = create_engine(f'sqlite:///{self.analysis_path}/metadata.db', echo=self.echo)
        SQLBase.metadata.create_all(engine)

        # Create a session
        self.session = Session(engine)

    def close_session(self):
        self.session.close()

        # Session attribute *needs* to be deleted, to prevent serialization error
        #  during multiprocess' pickling, because it contains weakrefs
        del self.session

    def add_animal(self, animal_id: str) -> Animal:
        return Animal.create(analysis=self, animal_id=animal_id)

    def animals(self, *args, **kwargs) -> EntityCollection:
        return Animal.filter(self, *args, **kwargs)

    def recordings(self, *args, **kwargs) -> EntityCollection:
        return Recording.filter(self, *args, **kwargs)

    def rois(self, *args, **kwargs) -> EntityCollection:
        return Roi.filter(self, *args, **kwargs)

    def phases(self, *args, **kwargs) -> EntityCollection:
        return Phase.filter(self, *args, **kwargs)

    def export(self, path: str):
        # TODO: add some data export and stuff
        with h5py.File(path, 'w') as f:
            for animal in self.animals():
                animal.export_to(f)

    def get_temp_path(self, path: str):
        temp_path = os.path.join(self.analysis_path, 'temp', path)
        if not os.path.exists(temp_path):
            # Avoid error if concurrent process already created it in meantime
            os.makedirs(temp_path, exist_ok=True)

        return temp_path


def open_analysis(analysis_path: str, mode=Mode.analyse, **kwargs) -> Analysis:
    meta_path = f'{analysis_path}/metadata.db'
    if not os.path.exists(meta_path):
        raise ValueError(f'Path {meta_path} not found')

    summary = Analysis(analysis_path, mode=mode, **kwargs)

    return summary
