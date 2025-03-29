"""
Parser for financial formulas using a context-free grammar.

This module provides functionality for parsing financial formulas defined in 
characteristic definition files. It defines the grammar rules and constructs
an abstract syntax tree (AST) that can be compiled and executed.
"""

import re
from typing import Dict, Any, List, Tuple, Union, Optional, Callable
import ast
from dataclasses import dataclass

# Define tokens for our grammar
class TokenType:
    NUMBER = 'NUMBER'
    IDENTIFIER = 'IDENTIFIER'
    OPERATOR = 'OPERATOR'
    FUNCTION = 'FUNCTION'
    LEFT_PAREN = 'LEFT_PAREN'
    RIGHT_PAREN = 'RIGHT_PAREN'
    COMMA = 'COMMA'
    STRING = 'STRING'
    EQUALS = 'EQUALS'
    COLON = 'COLON'

@dataclass
class Token:
    type: str
    value: str
    position: int

class FormulaParser:
    """
    Parser for financial formulas using a context-free grammar.
    
    This class implements a recursive descent parser for financial formulas
    defined in characteristic definition files.
    """
    
    def __init__(self):
        """
        Initialize the formula parser.
        """
        # Define the regex patterns for tokens
        self.token_patterns = [
            (TokenType.NUMBER, r'\d+(\.\d+)?'),
            (TokenType.IDENTIFIER, r'[a-zA-Z_][a-zA-Z0-9_]*'),
            (TokenType.OPERATOR, r'[\+\-\*/\^]'),
            (TokenType.LEFT_PAREN, r'\('),
            (TokenType.RIGHT_PAREN, r'\)'),
            (TokenType.COMMA, r','),
            (TokenType.STRING, r'"[^"]*"'),
            (TokenType.EQUALS, r'='),
            (TokenType.COLON, r':'),
        ]
        
        # Create a combined regex pattern
        self.token_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.token_patterns)
        self.token_regex = re.compile(self.token_regex)
        
    def tokenize(self, formula: str) -> List[Token]:
        """
        Convert a formula string into a list of tokens.
        
        Args:
            formula: The formula string to tokenize
            
        Returns:
            List of tokens
            
        Raises:
            SyntaxError: If the formula contains invalid syntax
        """
        tokens = []
        for match in re.finditer(self.token_regex, formula):
            token_type = match.lastgroup
            token_value = match.group(0)
            position = match.start()
            
            # Skip whitespace
            if token_type is None or token_value.isspace():
                continue
                
            tokens.append(Token(type=token_type, value=token_value, position=position))
            
        return tokens
    
    def parse(self, formula: str) -> ast.AST:
        """
        Parse a formula string into an abstract syntax tree.
        
        Args:
            formula: The formula string to parse
            
        Returns:
            Abstract syntax tree for the formula
            
        Raises:
            SyntaxError: If the formula contains invalid syntax
        """
        # For now, we'll use Python's ast module as a placeholder
        # In the future, we'll implement a proper parser for our grammar
        try:
            parsed = ast.parse(formula, mode='eval')
            return parsed
        except SyntaxError as e:
            raise SyntaxError(f"Invalid formula syntax: {e}")
    
    def extract_variables(self, ast_node: ast.AST) -> List[str]:
        """
        Extract variable names from an AST node.
        
        Args:
            ast_node: AST node to extract variables from
            
        Returns:
            List of variable names used in the formula
        """
        variables = []
        
        # This is a simplified implementation
        # In the future, we'll properly traverse the AST
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Name):
                variables.append(node.id)
                
        return variables
    
    def extract_functions(self, ast_node: ast.AST) -> List[str]:
        """
        Extract function names from an AST node.
        
        Args:
            ast_node: AST node to extract functions from
            
        Returns:
            List of function names used in the formula
        """
        functions = []
        
        # This is a simplified implementation
        # In the future, we'll properly traverse the AST
        for node in ast.walk(ast_node):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                functions.append(node.func.id)
                
        return functions

# Grammar definition for financial formulas
# This is a placeholder for a more comprehensive grammar in the future

# Grammar:
# formula ::= expression
# expression ::= term (('+' | '-') term)*
# term ::= factor (('*' | '/') factor)*
# factor ::= unary ('^' unary)*
# unary ::= '-' unary | primary
# primary ::= NUMBER | IDENTIFIER | STRING | function_call | '(' expression ')'
# function_call ::= IDENTIFIER '(' (expression (',' expression)*)? ')'

def parse_formula(formula: str) -> ast.AST:
    """
    Parse a formula string into an abstract syntax tree.
    
    Args:
        formula: The formula string to parse
        
    Returns:
        Abstract syntax tree for the formula
        
    Raises:
        SyntaxError: If the formula contains invalid syntax
    """
    parser = FormulaParser()
    return parser.parse(formula)

def extract_variables(ast_node: ast.AST) -> List[str]:
    """
    Extract variable names from an AST node.
    
    Args:
        ast_node: AST node to extract variables from
        
    Returns:
        List of variable names used in the formula
    """
    parser = FormulaParser()
    return parser.extract_variables(ast_node)

def extract_functions(ast_node: ast.AST) -> List[str]:
    """
    Extract function names from an AST node.
    
    Args:
        ast_node: AST node to extract functions from
        
    Returns:
        List of function names used in the formula
    """
    parser = FormulaParser()
    return parser.extract_functions(ast_node) 