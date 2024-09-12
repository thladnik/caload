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


def parse(*entity_types: List[Type]):

    flat = {}
    for _type in entity_types:
        parent = getattr(_type, 'parent_type', None)
        if parent is not None:
            parent = parent.__name__
        flat[_type.__name__] = parent

    def build_nested_dict(flat):
        # A function that recursively builds the nested dictionary
        def nest_key(key):
            # Initialize the dictionary for the current key
            nested = {}
            # Iterate over the flat dictionary to find keys that depend on the current key
            for k, v in flat.items():
                if v == key:
                    # Recursively build the nested structure for the child key
                    nested[k] = nest_key(k)
            return nested

        # Build the outer dictionary by finding the top-level keys (those with value None)
        nested_dict = {}
        for key, value in flat.items():
            if value is None:
                nested_dict[key] = nest_key(key)

        return nested_dict

    print(flat)
    pprint.pprint(build_nested_dict(flat))


if __name__ == '__main__':

    parse(Animal, Recording, RoiParam, Roi, Phase, PhaseParam, AnimalProps)
