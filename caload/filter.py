from __future__ import annotations

import re
from datetime import date, datetime
from operator import or_
from typing import Any, TYPE_CHECKING, Tuple, Type, Union

from sqlalchemy import BinaryExpression, and_, not_
from sqlalchemy.orm import Query, Session, aliased

from caload import utils
from caload.sqltables import *

if TYPE_CHECKING:
    from caload.analysis import Analysis
    from caload.entities import *

__all__ = ['get_entity_query_by_ids', 'get_entity_query_by_attributes']

import re


def tokenize(expression):
    # Define a regular expression for matching operands, operators, boolean values, and NOT
    token_pattern = r"""
        (?P<FLOAT>\b\d+\.\d+\b)                  # Float numbers
        |(?P<INTEGER>\b\d+\b)                    # Integer numbers
        |(?P<BOOLEAN>\bTrue\b|\bFalse\b)         # Boolean values
        |(?P<IDENTIFIER>\b[\w/]+\b)              # Identifiers (like signal1 or name1/subname1)
        |(?P<STRING>'[^']*')                     # Strings (like 'helloworld')
        |(?P<EXIST>\bEXIST\b)                    # EXIST operator
        |(?P<OPERATOR>>=|<=|==|!=|[<>])          # Comparison operators
        |(?P<LOGICAL>AND|OR|XOR|NOT)             # Logical operators including NOT
        |(?P<LPAREN>\()                          # Left parenthesis
        |(?P<RPAREN>\))                          # Right parenthesis
    """

    # Compile the regex
    token_regex = re.compile(token_pattern, re.VERBOSE)

    # Find all tokens in the expression
    tokens = []
    for match in token_regex.finditer(expression):
        token_type = match.lastgroup
        token_value = match.group(token_type)

        if token_type == 'FLOAT':
            token_value = float(token_value)  # Convert float string to float
        elif token_type == 'INTEGER':
            token_value = int(token_value)  # Convert integer string to int
        elif token_type == 'BOOLEAN':
            # Keep boolean values as is
            token_value = True if token_value == 'True' else False
        elif token_type == 'STRING':
            token_value = token_value.strip("'")  # Remove quotes from string

        tokens.append(token_value)

    return tokens


def parse_expression(tokens):
    """
    Recursively parse the tokens into a nested dictionary (AST). Handles nested parentheses and NOT operator.
    """

    def parse_parentheses(tokens):
        stack = []
        current_expr = []

        while tokens:
            token = tokens.pop(0)

            if token == '(':
                # Start a new group: recursively parse the sub-expression inside parentheses
                current_expr.append(parse_parentheses(tokens))
            elif token == ')':
                # Close current group: end of the sub-expression
                break
            elif token == 'NOT':
                # NOT is a unary operator, it should apply to the next operand or expression
                next_expr = tokens.pop(0)
                if next_expr == '(':
                    current_expr.append({'NOT': parse_parentheses(tokens)})
                else:
                    operand = [next_expr]
                    if tokens and tokens[0] in ('>=', '<=', '==', '!=', '>', '<'):
                        operator = tokens.pop(0)
                        right_operand = tokens.pop(0)
                        operand.extend([operator, right_operand])
                    current_expr.append({'NOT': operand})
            elif token in ('AND', 'OR', 'XOR'):
                # If we encounter AND/OR/XOR, push current expression onto the stack and store the operator
                if current_expr:
                    stack.append(current_expr)
                    current_expr = []
                stack.append(token)
            elif token == 'EXIST':
                # Handle the EXIST operator, which acts on the next operand
                next_operand = tokens.pop(0)
                current_expr.append({'EXIST': next_operand})
            else:
                # Handle comparisons and operands
                if current_expr and isinstance(current_expr[-1], dict) and 'comparison' in current_expr[-1]:
                    current_expr[-1]['comparison'].append(token)
                else:
                    current_expr.append({'operand': token})

        # At the end of the group or expression, add the current expression to the stack
        if current_expr:
            stack.append(current_expr)

        # Now combine everything in stack
        if len(stack) == 1:
            return stack[0]
        else:
            # Flatten the single nested list if it's just wrapping the result
            result = []
            while stack:
                element = stack.pop(0)
                if isinstance(element, list) and len(element) == 1:
                    result.append(element[0])
                else:
                    result.append(element)
            return result

    return parse_parentheses(tokens)


def parse_boolean_expression(expression):
    tokens = tokenize(expression)
    return parse_expression(tokens)


def get_entity_query_by_ids(analysis: Analysis,
                            base_table: Type[AnimalTable, RecordingTable, RoiTable, PhaseTable],
                            animal_id: str = None,
                            rec_date: Union[str, date, datetime] = None,
                            rec_id: str = None,
                            roi_id: int = None,
                            phase_id: int = None) -> Query:
    # Convert date
    rec_date = utils.parse_date(rec_date)

    # Create query
    query = analysis.session.query(base_table)

    # Join parents and filter by ID
    if base_table in (RoiTable, PhaseTable):
        if rec_date is not None or rec_id is not None:
            query = query.join(RecordingTable)

        if rec_date is not None:
            query = query.filter(RecordingTable.date == rec_date)
        if rec_id is not None:
            query = query.filter(RecordingTable.id == rec_id)

    if base_table in (RecordingTable, RoiTable, PhaseTable) and animal_id is not None:
        query = query.join(AnimalTable).filter(AnimalTable.id == animal_id)

    # Filter bottom entities
    if base_table == RoiTable and roi_id is not None:
        query = query.filter(RoiTable.id == roi_id)
    if base_table == PhaseTable and phase_id is not None:
        query = query.filter(PhaseTable.id == phase_id)

    return query


logical_operators = {
    'AND': and_,
    'OR': or_
}


def _generate_attribute_filters(entity_type: Type[Union[Animal, Recording, Roi, Phase]], session: Session,
                                astree: Union[dict, list]) -> Union[and_, or_, not_, BinaryExpression[bool]]:

    # Unnest extra brackets
    while isinstance(astree, list) and len(astree) == 1:
        astree = astree[0]

    # Handle lists (representing complex expressions)
    if isinstance(astree, list):

        # If 2nd item is a logical operatorto connect two operands
        if isinstance(astree[1], str) and astree[1] in logical_operators:
            operator = logical_operators[astree[1]]
            return operator(_generate_attribute_filters(entity_type, session, astree[0]),
                            _generate_attribute_filters(entity_type, session, astree[2:]))

        # Handle comparison expressions (like `attribute == value`)
        attribute_name = astree[0]['operand']
        comparator = astree[1]['operand']
        value = astree[2]['operand']

        # Determine the correct column based on value type
        attribute_value_col = getattr(entity_type.attr_value_table, f'value_{value.__class__.__name__}')

        # Build the comparison expression
        if comparator == '<':
            comparison = attribute_value_col < value
        elif comparator == '<=':
            comparison = attribute_value_col <= value
        elif comparator == '==':
            comparison = attribute_value_col == value
        elif comparator == '>=':
            comparison = attribute_value_col >= value
        elif comparator == '>':
            comparison = attribute_value_col > value
        else:
            raise ValueError(f"Unsupported comparator: {comparator}")

        # Build the subquery to filter entities matching the comparison
        subquery = session.query(entity_type.attr_value_table.entity_pk) \
            .join(AttributeTable) \
            .filter(AttributeTable.name == attribute_name, comparison) \
            .subquery()

    # Handle dictionaries (unary operators)
    elif isinstance(astree, dict):
        operator, astree = list(astree.items())[0]

        if operator == 'EXIST':
            subquery = session.query(entity_type.attr_value_table.entity_pk) \
                .join(AttributeTable) \
                .filter(AttributeTable.name == astree) \
                .subquery()

        elif operator == 'NOT':
            return not_(_generate_attribute_filters(entity_type, session, astree))

        else:
            print(f'Unkown unary operator: {operator}', astree)
            raise ValueError('Unexpected operator in the expression tree')

    else:
        print('Unexpected dictionary structure:', astree)
        raise ValueError('Unexpected structure in the expression tree')

    # Return the `IN` filter to apply to the main query
    return entity_type.entity_table.pk.in_(session.query(subquery.c.entity_pk))


def get_entity_query_by_attributes(entity_type: Type[Union[Animal, Recording, Roi, Phase]], session: Session,
                                   expr: str, entity_query: Query = None) -> Query:

    # Parse expression
    astree = parse_boolean_expression(expr)

    # Create base query
    query = session.query(entity_type.entity_table)

    # Filter for entities, if entity_query is provided
    if entity_query is not None:
        query = query.filter(entity_type.entity_table.pk.in_(entity_query.subquery().primary_key))

    # Apply filters generated from the abstract syntax tree (AST)
    filters = _generate_attribute_filters(entity_type, session, astree)
    query = query.filter(filters)

    return query
