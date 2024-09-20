import pprint
from typing import List, Type

from caload.entities import Entity


class Animal(Entity):
    pass


class Recording(Entity):

    parent_type = Animal


class Roi(Entity):

    parent_type = Recording


class Phase(Entity):

    parent_type = Recording


class PhaseParam(Entity):
    parent_type = Phase


class RoiParam(Entity):
    parent_type = Roi


class AnimalProps(Entity):

    parent_type = Animal





if __name__ == '__main__':

    parse(Animal, Recording, RoiParam, Roi, Phase, PhaseParam, AnimalProps)
