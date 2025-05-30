"""
Grammar definition for financial formulas.

This module defines the context-free grammar (CFG) used to parse financial formulas.
A CFG is formally defined as a 4-tuple G = (V, Σ, R, S) where:
- V is a finite set of non-terminal symbols
- Σ is a finite set of terminal symbols
- R is a finite set of production rules of the form A → α where A ∈ V and α ∈ (V ∪ Σ)*
- S is the start symbol, S ∈ V

This grammar supports:
- Basic arithmetic operations: +, -, *, /, ^ (exponentiation)
- Parenthesized expressions
- Function calls with multiple arguments
- Variables (identifiers)
- Numeric literals (integers and floating-point numbers)

The grammar follows standard mathematical precedence rules:
- Parentheses have highest precedence
- Exponentiation (^) is next
- Multiplication and division (* and /) have equal precedence
- Addition and subtraction (+ and -) have lowest precedence

Examples of valid formulas:
    2 + 3 * 4
    (x + y) / (z - 1)
    foo(x, y * z)
    -a + b^2 * c
"""

import re
from typing import Dict, List, Set, Tuple, FrozenSet, Optional, Match, Iterator, Union

# TODO: Add support for:
#   - Custom operator definitions
#   - Extended function syntax (named args, etc)
#   - Type annotations in formulas
#   - Error recovery during parsing

# Define the grammar components

# Set of non-terminal symbols
NON_TERMINALS: FrozenSet[str] = frozenset([
    'Expression',  # Top-level expressions (e.g., x + y)
    'Term',        # Terms in an expression (e.g., x * y)
    'Factor',      # Factors in a term (e.g., -x)
    'Power',       # Exponentiation (e.g., x^y)
    'Primary',     # Basic units (variables, literals, parenthesized expressions)
    'FunctionCall', # Function calls (e.g., foo(x, y))
    'Arguments',   # Function arguments (e.g., x, y, z)
    'Identifier',  # Variable and function names
    'Number',      # Numeric literals
    'DataVariable' # DataArray references (e.g., $close)
])

# Set of terminal symbols
TERMINALS: FrozenSet[str] = frozenset([
    '+', '-', '*', '/', '^',  # Operators
    '(', ')', ',',           # Delimiters
    # Identifiers and numbers are represented by regex patterns
    'IDENTIFIER',  # /[a-zA-Z_][a-zA-Z0-9_]*/
    'DATAVAR',     # /$[a-zA-Z_][a-zA-Z0-9_]*/
    'NUMBER',      # /[0-9]+(\.[0-9]+)?/
])

# Production rules defined as a dictionary from non-terminals to lists of productions
PRODUCTIONS: Dict[str, List[str]] = {
    # Expressions handle addition and subtraction
    'Expression': [
        'Term',                  # Simple term
        'Expression "+" Term',   # Addition
        'Expression "-" Term'    # Subtraction
    ],
    # Terms handle multiplication and division
    'Term': [
        'Factor',               # Simple factor
        'Term "*" Factor',      # Multiplication
        'Term "/" Factor'       # Division
    ],
    # Factors handle unary minus
    'Factor': [
        'Power',                # Simple power expression
        '"-" Factor'            # Negation
    ],
    # Power handles exponentiation
    'Power': [
        'Primary',              # Simple primary
        'Primary "^" Factor'    # Exponentiation
    ],
    # Primary elements are the basic building blocks
    'Primary': [
        'Number',               # Numeric literal
        'Identifier',           # Variable
        'DATAVAR',              # DataArray variable (with $ prefix)
        'FunctionCall',         # Function call
        '"(" Expression ")"'    # Parenthesized expression
    ],
    # Function calls with arguments
    'FunctionCall': [
        'Identifier "(" Arguments ")"'  # Function call with arguments
    ],
    # Arguments is a comma-separated list of expressions
    'Arguments': [
        'Expression',             # Single argument
        'Arguments "," Expression', # Multiple arguments
        ''                        # Empty arguments (epsilon)
    ],
    # Identifiers are variable and function names
    'Identifier': [
        'IDENTIFIER'              # Variable or function name
    ],
    # Numbers are numeric literals
    'Number': [
        'NUMBER'                  # Numeric literal
    ]
}

# Start symbol for the grammar
START_SYMBOL: str = 'Expression'

# The complete CFG 4-tuple
FINANCIAL_FORMULA_GRAMMAR: Tuple[FrozenSet[str], FrozenSet[str], Dict[str, List[str]], str] = (
    NON_TERMINALS,
    TERMINALS,
    PRODUCTIONS,
    START_SYMBOL
)

# Regular expressions for lexical analysis
REGEX_PATTERNS = {
    'IDENTIFIER': r'[a-zA-Z_][a-zA-Z0-9_]*',  # Variables and function names
    'DATAVAR': r'\$[a-zA-Z_][a-zA-Z0-9_]*',   # DataArray variables with $ prefix
    'NUMBER': r'([0-9]+(\.[0-9]+)?)|(\.[0-9]+)',  # Integer or decimal numbers
    'WHITESPACE': r'\s+',                      # Spaces, tabs, newlines
    'OPERATOR': r'[\+\-\*/\^]',                # Arithmetic operators
    'LPAREN': r'\(',                           # Left parenthesis
    'RPAREN': r'\)',                           # Right parenthesis
    'COMMA': r','                              # Comma for function arguments
}

# Operator precedence (higher number = higher precedence)
OPERATOR_PRECEDENCE = {
    '+': 1,
    '-': 1,
    '*': 2,
    '/': 2,
    '^': 3
}

# Operator associativity
OPERATOR_ASSOCIATIVITY = {
    '+': 'left',
    '-': 'left',
    '*': 'left',
    '/': 'left',
    '^': 'right'  # Exponentiation is right-associative: 2^3^2 = 2^(3^2)
}

def get_grammar_description() -> str:
    """
    Returns a human-readable description of the grammar.
    
    This function generates a detailed textual description of the context-free
    grammar, including all non-terminals, terminals, production rules, and the
    start symbol.
    
    Returns:
        A string containing the grammar description
        
    Examples:
        >>> print(get_grammar_description())
        Financial Formula Grammar (CFG):
        
        Non-terminals (V):
        Arguments, Expression, Factor, FunctionCall, Identifier, Number, Power, Primary, Term
        
        Terminals (Σ):
        (, ), *, +, ,, -, /, IDENTIFIER, NUMBER, ^
        
        Production Rules (R):
        Expression → Term
        Expression → Expression "+" Term
        Expression → Expression "-" Term
        ...
    """
    description = "Financial Formula Grammar (CFG):\n\n"
    
    description += "Non-terminals (V):\n"
    description += ", ".join(sorted(NON_TERMINALS)) + "\n\n"
    
    description += "Terminals (Σ):\n"
    description += ", ".join(sorted(TERMINALS)) + "\n\n"
    
    description += "Production Rules (R):\n"
    for nt, productions in PRODUCTIONS.items():
        for prod in productions:
            if prod:
                description += f"{nt} → {prod}\n"
            else:
                description += f"{nt} → ε\n"  # Empty string (epsilon)
    
    description += f"\nStart Symbol (S): {START_SYMBOL}\n"
    
    # Add precedence and associativity information
    description += "\nOperator Precedence (higher number = higher precedence):\n"
    for op, prec in sorted(OPERATOR_PRECEDENCE.items(), key=lambda x: x[1]):
        description += f"{op}: {prec}\n"
    
    description += "\nOperator Associativity:\n"
    for op, assoc in sorted(OPERATOR_ASSOCIATIVITY.items()):
        description += f"{op}: {assoc}\n"
    
    return description

def tokenize(formula: str) -> List[Tuple[str, str]]:
    """
    Tokenize a formula string into a list of tokens.
    
    This function breaks a formula string into tokens, where each token is a
    tuple of (token_type, token_value). The token types are:
    - 'IDENTIFIER': Variable and function names
    - 'DATAVAR': DataArray references with $ prefix 
    - 'NUMBER': Numeric literals
    - 'OPERATOR': Arithmetic operators (+, -, *, /, ^)
    - 'LPAREN': Left parenthesis
    - 'RPAREN': Right parenthesis
    - 'COMMA': Comma for function arguments
    
    Args:
        formula: The formula string to tokenize
        
    Returns:
        A list of (token_type, token_value) tuples
        
    Raises:
        ValueError: If an invalid token is found in the formula
        
    Examples:
        >>> tokenize("x + y * z")
        [('IDENTIFIER', 'x'), ('OPERATOR', '+'), ('IDENTIFIER', 'y'), ('OPERATOR', '*'), ('IDENTIFIER', 'z')]
        >>> tokenize("foo(x, 2.5)")
        [('IDENTIFIER', 'foo'), ('LPAREN', '('), ('IDENTIFIER', 'x'), ('COMMA', ','), ('NUMBER', '2.5'), ('RPAREN', ')')]
        >>> tokenize("$close / $volume")
        [('DATAVAR', '$close'), ('OPERATOR', '/'), ('DATAVAR', '$volume')]
    """
    tokens = []
    pos = 0
    
    # Define a pattern for all token types
    token_pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in REGEX_PATTERNS.items())
    token_regex = re.compile(token_pattern)
    
    while pos < len(formula):
        match = token_regex.match(formula, pos)
        if not match:
            # No valid token found, raise an error
            raise ValueError(f"Invalid token at position {pos}: '{formula[pos:]}'")
        
        pos = match.end()
        
        # Skip whitespace
        if match.lastgroup == 'WHITESPACE':
            continue
        
        # Get the token type and value
        token_type = match.lastgroup
        token_value = match.group()
        
        # Determine the correct type for operators and delimiters
        if token_type == 'OPERATOR':
            tokens.append((token_value, token_value))  # Use the operator as both type and value
        elif token_type == 'LPAREN':
            tokens.append(('(', token_value))
        elif token_type == 'RPAREN':
            tokens.append((')', token_value))
        elif token_type == 'COMMA':
            tokens.append((',', token_value))
        elif token_type == 'DATAVAR':
            tokens.append((token_type, token_value))  # Keep the $ in the token value
        else:
            tokens.append((token_type, token_value))
    
    return tokens

def check_formula_syntax(formula: str) -> Tuple[bool, Optional[str]]:
    """
    Check if a formula has valid syntax according to the grammar.
    
    This function performs a basic check of formula syntax by tokenizing
    the formula and looking for common syntax errors. It does not perform
    a full parse of the formula.
    
    Args:
        formula: The formula string to check
        
    Returns:
        A tuple of (is_valid, error_message) where is_valid is a boolean and
        error_message is None if valid or a string describing the error
        
    Examples:
        >>> check_formula_syntax("x + y * z")
        (True, None)
        >>> check_formula_syntax("x + * y")
        (False, "Syntax error: unexpected operator '*' after operator '+'")
    """
    try:
        tokens = tokenize(formula)
    except ValueError as e:
        return False, str(e)
    
    # Basic syntax checking
    
    # Check for empty formula
    if not tokens:
        return False, "Empty formula"
    
    # Check for balanced parentheses
    paren_count = 0
    for token_type, _ in tokens:
        if token_type == '(':
            paren_count += 1
        elif token_type == ')':
            paren_count -= 1
            if paren_count < 0:
                return False, "Unbalanced parentheses: too many closing parentheses"
    
    if paren_count > 0:
        return False, "Unbalanced parentheses: missing closing parenthesis"
    
    # Check for adjacent operators and other basic syntax rules
    for i in range(len(tokens) - 1):
        curr_type, curr_val = tokens[i]
        next_type, next_val = tokens[i + 1]
        
        # Can't have two operators in a row
        if curr_type in OPERATOR_PRECEDENCE and next_type in OPERATOR_PRECEDENCE:
            return False, f"Syntax error: unexpected operator '{next_val}' after operator '{curr_val}'"
        
        # Can't have identifier followed by identifier
        if curr_type == 'IDENTIFIER' and next_type == 'IDENTIFIER':
            return False, f"Syntax error: unexpected identifier '{next_val}' after identifier '{curr_val}'"
        
        # Can't have number followed by identifier
        if curr_type == 'NUMBER' and next_type == 'IDENTIFIER':
            return False, f"Syntax error: unexpected identifier '{next_val}' after number '{curr_val}'"
        
        # Can't have identifier followed by number
        if curr_type == 'IDENTIFIER' and next_type == 'NUMBER':
            return False, f"Syntax error: unexpected number '{next_val}' after identifier '{curr_val}'"
        
        # Can't have closing parenthesis followed by identifier or number
        if curr_type == ')' and next_type in ('IDENTIFIER', 'NUMBER'):
            return False, f"Syntax error: unexpected {next_type.lower()} '{next_val}' after closing parenthesis"
        
        # Can't have number or identifier followed by opening parenthesis (except for function calls)
        if curr_type == 'NUMBER' and next_type == '(':
            return False, f"Syntax error: unexpected opening parenthesis after number '{curr_val}'"
    
    # Check for starting with an operator (except unary minus)
    if tokens[0][0] in ('+', '*', '/', '^'):
        return False, f"Syntax error: formula can't start with operator '{tokens[0][1]}'"
    
    # Check for ending with an operator
    if tokens[-1][0] in OPERATOR_PRECEDENCE:
        return False, f"Syntax error: formula can't end with operator '{tokens[-1][1]}'"
    
    # If we get here, the formula passed all basic syntax checks
    return True, None

def get_operator_precedence(operator: str) -> int:
    """
    Get the precedence of an operator.
    
    Higher number means higher precedence.
    
    Args:
        operator: The operator (+, -, *, /, ^)
        
    Returns:
        The precedence as an integer
        
    Raises:
        ValueError: If the operator is not recognized
        
    Examples:
        >>> get_operator_precedence('+')
        1
        >>> get_operator_precedence('*')
        2
        >>> get_operator_precedence('^')
        3
    """
    if operator not in OPERATOR_PRECEDENCE:
        raise ValueError(f"Unknown operator: {operator}")
    return OPERATOR_PRECEDENCE[operator]

def is_right_associative(operator: str) -> bool:
    """
    Check if an operator is right-associative.
    
    Args:
        operator: The operator (+, -, *, /, ^)
        
    Returns:
        True if the operator is right-associative, False otherwise
        
    Raises:
        ValueError: If the operator is not recognized
        
    Examples:
        >>> is_right_associative('+')
        False
        >>> is_right_associative('^')
        True
    """
    if operator not in OPERATOR_ASSOCIATIVITY:
        raise ValueError(f"Unknown operator: {operator}")
    return OPERATOR_ASSOCIATIVITY[operator] == 'right'

def get_example_formulas() -> List[str]:
    """
    Get a list of example formulas that conform to the grammar.
    
    These examples demonstrate various features of the grammar and
    can be used for testing or documentation purposes.
    
    Returns:
        A list of example formula strings
        
    Examples:
        >>> examples = get_example_formulas()
        >>> len(examples) > 0
        True
        >>> "x + y * z" in examples
        True
    """
    return [
        # Basic arithmetic
        "2 + 3",
        "2 - 3",
        "2 * 3",
        "6 / 3",
        "2 ^ 3",
        
        # Combined operations
        "2 + 3 * 4",
        "(2 + 3) * 4",
        "2 * 3 + 4",
        "2 - 3 * 4",
        
        # Variables
        "x",
        "x + y",
        "x * y + z",
        "x / (y + z)",
        
        # Unary minus
        "-x",
        "-(x + y)",
        "-x * y",
        
        # Function calls
        "f(x)",
        "f(x, y)",
        "f(x + y, z * w)",
        "f(g(x), h(y, z))",
        
        # Complex expressions
        "a + b * c ^ d / e - f",
        "(a + b) * (c - d) / (e ^ f)",
        "f(x, y) + g(a * b, c / d)"
    ] 