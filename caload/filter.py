from __future__ import annotations

import pprint
import re
from datetime import date, datetime
from typing import Any, Dict, TYPE_CHECKING, Tuple, Type, Union

from sqlalchemy import BinaryExpression, or_, and_, not_
from sqlalchemy.orm import Query, Session, aliased

from caload import utils
from caload.sqltables import *

if TYPE_CHECKING:
    from caload.analysis import Analysis
    from caload.entities import *

__all__ = ['get_entity_query_by_attributes']


def tokenize(expression):
    # Define a regular expression for matching operands, operators, boolean values, dates, datetimes, and IN/NOT
    token_pattern = r"""
        (?P<STRING_SINGLE>'[^']*')                               # Strings in single quotes
        |(?P<STRING_DOUBLE>"[^"]*")                              # Strings in double quotes
        |(?P<DATETIME>\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\b)    # ISO 8601 datetime (YYYY-MM-DDTHH:MM:SS)
        |(?P<DATE>\b\d{4}-\d{2}-\d{2}\b)                         # Date (YYYY-MM-DD)
        |(?P<FLOAT>\b\d+\.\d+\b)                                 # Float numbers
        |(?P<INTEGER>\b\d+\b)                                    # Integer numbers
        |(?P<BOOLEAN>\bTrue\b|\bFalse\b)                         # Boolean values
        |(?P<IDENTIFIER>\b[\w/]+\b)                              # Identifiers (like signal1 or name1/subname1)
        |(?P<EXIST>\bEXIST\b)                                    # EXIST operator
        |(?P<IN>\bIN\b)                                          # IN operator
        |(?P<OPERATOR>>=|<=|==|!=|[<>])                          # Comparison operators
        |(?P<LOGICAL>AND|OR|XOR|NOT)                             # Logical operators including NOT
        |(?P<LPAREN>\()                                          # Left parenthesis
        |(?P<RPAREN>\))                                          # Right parenthesis
    """

    # Compile the regex
    token_regex = re.compile(token_pattern, re.VERBOSE | re.IGNORECASE)

    # Find all tokens in the expression
    tokens = []
    for match in token_regex.finditer(expression):
        token_type = match.lastgroup
        token_value = match.group(token_type)

        if token_type == 'DATETIME':
            # Convert datetime string to datetime.datetime object
            token_value = datetime.strptime(token_value, '%Y-%m-%dT%H:%M:%S')
        elif token_type == 'DATE':
            # Convert date string to datetime.date object only if it's not enclosed in quotes
            token_value = datetime.strptime(token_value, '%Y-%m-%d').date()
        elif token_type == 'FLOAT':
            token_value = float(token_value)  # Convert float string to float
        elif token_type == 'INTEGER':
            token_value = int(token_value)  # Convert integer string to int
        elif token_type == 'BOOLEAN':
            # Keep boolean values as is
            token_value = True if token_value == 'True' else False
        elif token_type == 'STRING_SINGLE' or token_type == 'STRING_DOUBLE':
            # Strip the quotes around the string and treat it as a string
            token_value = token_value[1:-1]

        tokens.append(token_value)

    return tokens


def parse_expression(tokens):
    """
    Recursively parse the tokens into a flattened nested dictionary structure (AST).
    Binary operations have 'left_operand', 'operator', and 'right_operand'.
    Unary operations have 'operator' and 'right_operand'.
    """

    def parse_parentheses(tokens):
        current_expr = None
        while tokens:
            token = tokens.pop(0)

            if token == '(':
                # Start a new group: recursively parse the sub-expression inside parentheses
                sub_expr = parse_parentheses(tokens)
                current_expr = merge_expression(current_expr, sub_expr)
            elif token == ')':
                # Close current group: end of the sub-expression
                break
            elif token.upper() == 'NOT':
                # NOT is a unary operator, it should apply to the next operand or expression
                next_expr = tokens.pop(0)
                if next_expr == '(':
                    parsed_expr = parse_parentheses(tokens)
                else:
                    parsed_expr = next_expr
                current_expr = merge_expression(current_expr, {
                    'operator': 'NOT',
                    'right_operand': parsed_expr
                })
            elif token.upper() in ('AND', 'OR', 'XOR'):
                # Handle AND/OR/XOR operators between expressions
                current_expr = {
                    'left_operand': current_expr,
                    'operator': token,
                    'right_operand': parse_parentheses(tokens)
                }
            elif token.upper() == 'EXIST':
                # Handle the EXIST operator, which acts on the next operand
                next_operand = tokens.pop(0)
                if next_operand == '(':
                    next_operand = parse_parentheses(tokens)
                current_expr = merge_expression(current_expr, {
                    'operator': 'EXIST',
                    'right_operand': next_operand
                })
            elif token.upper() == 'IN':
                # Handle the IN operator, which checks if the left operand is in the right list
                left_operand = current_expr
                if tokens[0] == '(':  # Expecting a list enclosed in parentheses
                    tokens.pop(0)  # Remove the '('
                    right_operand = []
                    while tokens[0] != ')':  # Collect all values inside the parentheses
                        right_operand.append(tokens.pop(0))
                    tokens.pop(0)  # Remove the closing ')'
                else:
                    right_operand = tokens.pop(0)
                current_expr = {
                    'left_operand': left_operand,
                    'operator': 'IN',
                    'right_operand': right_operand
                }
            else:
                # Handle comparisons and other binary operations
                if current_expr:
                    left_operand = current_expr
                    operator = token
                    right_operand = tokens.pop(0)
                    current_expr = {
                        'left_operand': left_operand,
                        'operator': operator,
                        'right_operand': right_operand
                    }
                else:
                    current_expr = token

        return current_expr

    def merge_expression(left_expr, right_expr):
        """
        Merges two expressions, ensuring they are combined without unnecessary nesting.
        """
        if left_expr is None:
            return right_expr
        return {
            'left_operand': left_expr,
            'operator': 'AND',  # Default to 'AND' if the operator is implicit
            'right_operand': right_expr
        }

    return parse_parentheses(tokens)


logical_operators = {
    'AND': and_,
    'OR': or_
}


def _generate_attribute_filters(entity_type_name: str, session: Session,
                                astree: Dict[str, Any]) -> Union[and_, or_, not_, BinaryExpression[bool]]:
    """Generate SQLAlchemy ORM filters given a filter syntax tree
    TODO: currently supported operations are nested trees of: AND, OR, IN, EXIST, NOT, <=, <, ==, >, >=
    """

    operator = astree['operator'].upper()

    # Handle connectives
    if operator in ('AND', 'OR'):
        op = logical_operators[operator]
        return op(_generate_attribute_filters(entity_type_name, session, astree['left_operand']),
                  _generate_attribute_filters(entity_type_name, session, astree['right_operand']))

    # Handle comparisons
    elif operator in ('IN', '<=', '<', '==', '>', '>='):

        name = astree['left_operand']
        value = astree['right_operand']

        if operator == 'IN':
            if not isinstance(value, list):
                raise ValueError('Operand after IN statement should be a list of values')

            attribute_value_col = getattr(AttributeTable, f'value_{value[0].__class__.__name__}')

            comparison = attribute_value_col.in_(value)
        else:
            # Determine the correct column based on value type
            attribute_value_col = getattr(AttributeTable, f'value_{value.__class__.__name__}')

            # Build the comparison expression
            if operator == '<':
                comparison = attribute_value_col < value
            elif operator == '<=':
                comparison = attribute_value_col <= value
            elif operator == '==':
                comparison = attribute_value_col == value
            elif operator == '>=':
                comparison = attribute_value_col >= value
            elif operator == '>':
                comparison = attribute_value_col > value
            else:
                raise ValueError(f"Unsupported comparator: {operator}")

        # Build the subquery to filter entities matching the comparison
        subquery = (session.query(AttributeTable.entity_pk).filter(AttributeTable.name == name, comparison).join(EntityTable)
                    .join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)
                    .subquery())

    # Handle unary operators
    elif operator == 'EXIST':
        subquery = session.query(AttributeTable.entity_pk) \
            .filter(AttributeTable.name == astree['right_operand']) \
            .subquery()

    elif operator == 'NOT':
        return not_(_generate_attribute_filters(entity_type_name, session, astree['right_operand']))

    # Fallback
    else:
        print(f'Unkown unary operator: {operator}', astree)
        raise ValueError('Unexpected operator in the expression tree')

    # Return the `IN` filter to apply to the main query
    return EntityTable.pk.in_(session.query(subquery.c.entity_pk))


def parse_boolean_expression(expression):
    tokens = tokenize(expression)
    return parse_expression(tokens)


def get_entity_query_by_attributes(entity_type_name: str, session: Session,
                                   expr: str, entity_query: Query = None) -> Query:
    # Create base query
    query = session.query(EntityTable).join(EntityTypeTable).filter(EntityTypeTable.name == entity_type_name)

    # Filter for entities, if entity_query is provided
    if entity_query is not None:
        query = query.filter(EntityTable.pk.in_(entity_query.subquery().primary_key))

    # Parse expression
    if expr != '':
        # Parse expression into syntax tree
        astree = parse_boolean_expression(expr)
        # pprint.pprint(astree)

        # Apply filters generated from the abstract syntax tree (AST)
        filters = _generate_attribute_filters(entity_type_name, session, astree)
        query = query.filter(filters)

    return query


if __name__ == '__main__':
    # expression = "(signal1 == 2024-09-06 AND signal2 > 2024-09-06T14:30:00 OR signal3 == '2024-08-02_fish1')"
    # expression = "(animal_id == 2024-08-02_fish1)"
    # expression = 'rec_id == "rec2" AND rec_date == 2024-07-07 AND animal_id == "2024-08-02_fish1"'
    expression_string = ("name IN (1.2, 10, 'Charlie') "
                         "AND "
                         "(EXIST signal1) "
                         "AND rec_id == 'rec2' "
                         "AND NOT (rec_date == 2024-07-07 OR animal_id == '2024-08-02_fish1')")

    parsed_expression = parse_boolean_expression(expression_string)

    pprint.pprint(parsed_expression)
