"""
Parser for financial formulas using a context-free grammar.

This module provides a comprehensive parser for financial formulas based on a context-free grammar.
It converts formula strings into Abstract Syntax Tree (AST) nodes that can be evaluated with JAX.

The parser handles common mathematical operations (addition, subtraction, multiplication, division,
exponentiation), variables, numeric literals, and function calls with arguments. It provides detailed
error reporting and dependency analysis capabilities.

Examples:
    Basic parsing:
    >>> from src.data.ast.parser import parse_formula
    >>> ast = parse_formula("2 + 3 * 4")
    >>> ast.evaluate({})
    14.0
    
    Parsing with variables:
    >>> ast = parse_formula("x + y * z")
    >>> ast.evaluate({'x': 1.0, 'y': 2.0, 'z': 3.0})
    7.0
    
    Parsing function calls:
    >>> ast = parse_formula("add(x, y)")
    >>> ast.evaluate({'x': 1.0, 'y': 2.0, 'add': lambda a, b: a + b})
    3.0
"""

import ast
import re
from typing import Dict, Any, List, Union, Optional, Set, Tuple, cast

from .nodes import Node, BinaryOp, UnaryOp, Literal, Variable, FunctionCall
from .grammar import FINANCIAL_FORMULA_GRAMMAR

class FormulaParser:
    """
    A comprehensive parser for financial formulas.
    
    This parser uses Python's built-in ast module to parse the formula and then
    converts the Python AST to our custom AST nodes. It supports the standard
    mathematical operations, variables, function calls, and nested expressions.
    
    The parser provides detailed error reporting and can extract information about
    formula dependencies (variables and functions used).
    
    Attributes:
        _cache: A cache of parsed formulas to avoid redundant parsing
    """
    
    def __init__(self):
        """Initialize a new FormulaParser with an empty cache."""
        self._cache: Dict[str, Node] = {}
    
    def parse(self, formula: str, use_cache: bool = True) -> Node:
        """
        Parse a formula string into a Node.
        
        This method converts a string representation of a formula into an Abstract
        Syntax Tree (AST) represented by Node objects. The resulting AST can be
        evaluated with different variable values.
        
        Args:
            formula: The formula string to parse
            use_cache: Whether to use the parser's cache for previously parsed formulas
            
        Returns:
            The root Node of the AST
            
        Raises:
            SyntaxError: If the formula contains invalid syntax
            ValueError: If the formula contains unsupported operations or functions
            
        Examples:
            >>> parser = FormulaParser()
            >>> ast = parser.parse("2 + 3")
            >>> ast.evaluate({})
            5.0
        """
        # Check cache first if enabled
        if use_cache and formula in self._cache:
            return self._cache[formula]
        
        try:
            # Replace caret with ** for Python's power operator
            python_formula = formula.replace('^', '**')
            
            # Parse the formula using Python's ast module
            py_ast = ast.parse(python_formula, mode='eval')
            
            # Convert the Python AST to our custom AST
            node = self._convert_node(py_ast.body)
            
            # Store in cache if enabled
            if use_cache:
                self._cache[formula] = node
                
            return node
        except SyntaxError as e:
            line_info = f" at line {e.lineno}, column {e.offset}" if hasattr(e, 'lineno') and hasattr(e, 'offset') else ""
            raise SyntaxError(f"Invalid formula syntax{line_info}: {e.msg}\nFormula: {formula}")
        except Exception as e:
            # Catch any other errors and provide context
            raise ValueError(f"Error parsing formula '{formula}': {str(e)}")
    
    def clear_cache(self):
        """
        Clear the parser's formula cache.
        
        This can be useful for freeing memory or if you need to re-parse formulas
        that might have been cached with different behavior.
        """
        self._cache = {}
    
    def _convert_node(self, node: ast.AST) -> Node:
        """
        Convert a Python AST node to our custom AST node.
        
        This internal method recursively converts Python's AST nodes to our
        custom Node subclasses that can be evaluated with JAX.
        
        Args:
            node: Python AST node
            
        Returns:
            Our custom AST node
            
        Raises:
            ValueError: If the node type is not supported
        """
        # Handle different node types
        if isinstance(node, ast.BinOp):
            # Binary operation
            op_map = {
                ast.Add: '+',
                ast.Sub: '-',
                ast.Mult: '*',
                ast.Div: '/',
                ast.Pow: '^'
            }
            op = op_map.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported binary operator: {node.op}")
            
            left = self._convert_node(node.left)
            right = self._convert_node(node.right)
            
            return BinaryOp(op=op, left=left, right=right)
        
        elif isinstance(node, ast.UnaryOp):
            # Unary operation
            op_map = {
                ast.USub: '-',
                ast.UAdd: '+'  # This is a no-op
            }
            op = op_map.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operator: {node.op}")
            
            if op == '+':
                # Unary + is a no-op, just return the operand
                return self._convert_node(node.operand)
            
            operand = self._convert_node(node.operand)
            
            return UnaryOp(op=op, operand=operand)
        
        elif isinstance(node, ast.Num):
            # Literal number (Python 3.7 and earlier)
            return Literal(value=float(node.n))
        
        elif isinstance(node, ast.Constant):
            # Literal constant (Python 3.8+)
            if isinstance(node.value, (int, float)):
                return Literal(value=float(node.value))
            elif node.value is None:
                raise ValueError("Null values are not supported in formulas")
            else:
                raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
        
        elif isinstance(node, ast.Name):
            # Variable
            return Variable(name=node.id)
        
        elif isinstance(node, ast.Call):
            # Function call
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                args = [self._convert_node(arg) for arg in node.args]
                
                # Check for keyword arguments, which we don't support
                if node.keywords:
                    kw_args = [f"{kw.arg}={self._node_to_str(kw.value)}" for kw in node.keywords]
                    raise ValueError(f"Keyword arguments are not supported: {', '.join(kw_args)}")
                
                return FunctionCall(name=func_name, args=args)
            else:
                # This would catch things like (x+y)(z) which is not valid in our grammar
                raise ValueError(f"Unsupported function call: {ast.dump(node.func)}")
        
        else:
            # Unsupported node type
            raise ValueError(f"Unsupported expression type: {type(node).__name__}")
    
    def _node_to_str(self, node: ast.AST) -> str:
        """Convert a Python AST node back to a string representation for error messages."""
        if isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                args = [self._node_to_str(arg) for arg in node.args]
                return f"{node.func.id}({', '.join(args)})"
        return "<complex expression>"
    
    def extract_variables(self, node: Node) -> Set[str]:
        """
        Extract all variable names from the AST.
        
        This method recursively traverses the AST and collects all the variables
        used in the formula. This can be useful for dependency tracking or validation.
        
        Args:
            node: Root node of the AST
            
        Returns:
            Set of variable names used in the formula
            
        Examples:
            >>> parser = FormulaParser()
            >>> ast = parser.parse("x + y * z")
            >>> parser.extract_variables(ast)
            {'x', 'y', 'z'}
        """
        variables = set()
        
        def _extract(n):
            if isinstance(n, Variable):
                variables.add(n.name)
            elif isinstance(n, BinaryOp):
                _extract(n.left)
                _extract(n.right)
            elif isinstance(n, UnaryOp):
                _extract(n.operand)
            elif isinstance(n, FunctionCall):
                for arg in n.args:
                    _extract(arg)
        
        _extract(node)
        return variables
    
    def extract_functions(self, node: Node) -> Set[str]:
        """
        Extract all function names from the AST.
        
        This method recursively traverses the AST and collects all the function names
        used in the formula. This can be useful for dependency tracking or validation.
        
        Args:
            node: Root node of the AST
            
        Returns:
            Set of function names used in the formula
            
        Examples:
            >>> parser = FormulaParser()
            >>> ast = parser.parse("add(x, multiply(y, z))")
            >>> parser.extract_functions(ast)
            {'add', 'multiply'}
        """
        functions = set()
        
        def _extract(n):
            if isinstance(n, FunctionCall):
                functions.add(n.name)
                for arg in n.args:
                    _extract(arg)
            elif isinstance(n, BinaryOp):
                _extract(n.left)
                _extract(n.right)
            elif isinstance(n, UnaryOp):
                _extract(n.operand)
        
        _extract(node)
        return functions
    
    def validate_against_grammar(self, formula: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a formula conforms to the defined grammar.
        
        This method checks if a formula follows the context-free grammar
        defined for financial formulas. This is a more strict check than
        just parsing the formula.
        
        Note: Currently implemented as a basic parsing check. A more sophisticated
        validation based on the formal grammar rules could be implemented in the future.
        
        Args:
            formula: The formula string to validate
            
        Returns:
            A tuple of (is_valid, error_message) where is_valid is a boolean and
            error_message is None if valid or a string describing the error
            
        Examples:
            >>> parser = FormulaParser()
            >>> valid, error = parser.validate_against_grammar("x + y * z")
            >>> valid
            True
            >>> valid, error = parser.validate_against_grammar("x +* y")
            >>> valid
            False
            >>> error
            'Invalid formula syntax: unexpected token "*" after "+"'
        """
        try:
            # Try to parse the formula - if it succeeds, it's valid according to our grammar
            self.parse(formula, use_cache=False)
            return True, None
        except (SyntaxError, ValueError) as e:
            return False, str(e)
        except Exception as e:
            return False, f"Unexpected error validating formula: {str(e)}"
    
    def optimize_ast(self, node: Node) -> Node:
        """
        Optimize an AST by applying various transformations.
        
        This method applies optimizations like constant folding to the AST.
        For example, the expression "2 + 3" would be optimized to a single
        Literal node with value 5.
        
        Args:
            node: The root node of the AST to optimize
            
        Returns:
            An optimized version of the AST
            
        Examples:
            >>> parser = FormulaParser()
            >>> ast = parser.parse("2 + 3")
            >>> optimized = parser.optimize_ast(ast)
            >>> isinstance(optimized, Literal)
            True
            >>> optimized.value
            5.0
        """
        # Basic constant folding
        if isinstance(node, BinaryOp):
            # Recursively optimize left and right
            left = self.optimize_ast(node.left)
            right = self.optimize_ast(node.right)
            
            # If both sides are literals, compute the result
            if isinstance(left, Literal) and isinstance(right, Literal):
                try:
                    if node.op == '+':
                        return Literal(value=left.value + right.value)
                    elif node.op == '-':
                        return Literal(value=left.value - right.value)
                    elif node.op == '*':
                        return Literal(value=left.value * right.value)
                    elif node.op == '/':
                        if right.value == 0:
                            # Avoid division by zero
                            return BinaryOp(op=node.op, left=left, right=right)
                        return Literal(value=left.value / right.value)
                    elif node.op == '^':
                        return Literal(value=left.value ** right.value)
                except Exception:
                    # If any error occurs during folding, return the original structure
                    pass
            
            # If optimization didn't result in a literal, return a new BinaryOp with optimized children
            return BinaryOp(op=node.op, left=left, right=right)
        
        elif isinstance(node, UnaryOp):
            # Recursively optimize the operand
            operand = self.optimize_ast(node.operand)
            
            # If operand is a literal, compute the result
            if isinstance(operand, Literal):
                if node.op == '-':
                    return Literal(value=-operand.value)
            
            # If optimization didn't result in a literal, return a new UnaryOp with optimized operand
            return UnaryOp(op=node.op, operand=operand)
        
        elif isinstance(node, FunctionCall):
            # Recursively optimize all arguments
            optimized_args = [self.optimize_ast(arg) for arg in node.args]
            
            # Functions can't be computed at optimization time since they're not known
            # So just return a new FunctionCall with optimized arguments
            return FunctionCall(name=node.name, args=optimized_args)
        
        # Variables and Literals don't need optimization
        return node

# Create a singleton parser instance
_parser = FormulaParser()

def parse_formula(formula: str) -> Node:
    """
    Parse a formula string into a Node.
    
    This is a convenience function that uses the singleton parser instance.
    
    Args:
        formula: The formula string to parse
        
    Returns:
        The root Node of the AST
        
    Examples:
        >>> ast = parse_formula("x + y * z")
        >>> ast.evaluate({'x': 1.0, 'y': 2.0, 'z': 3.0})
        7.0
    """
    return _parser.parse(formula)

def extract_variables(formula: Union[str, Node]) -> Set[str]:
    """
    Extract all variable names from a formula.
    
    This is a convenience function that uses the singleton parser instance.
    
    Args:
        formula: Formula string or AST node
        
    Returns:
        Set of variable names used in the formula
        
    Examples:
        >>> extract_variables("x + y * z")
        {'x', 'y', 'z'}
    """
    if isinstance(formula, str):
        node = parse_formula(formula)
    else:
        node = formula
    
    return _parser.extract_variables(node)

def extract_functions(formula: Union[str, Node]) -> Set[str]:
    """
    Extract all function names from a formula.
    
    This is a convenience function that uses the singleton parser instance.
    
    Args:
        formula: Formula string or AST node
        
    Returns:
        Set of function names used in the formula
        
    Examples:
        >>> extract_functions("add(x, multiply(y, z))")
        {'add', 'multiply'}
    """
    if isinstance(formula, str):
        node = parse_formula(formula)
    else:
        node = formula
    
    return _parser.extract_functions(node)

def optimize_formula(formula: Union[str, Node]) -> Node:
    """
    Optimize a formula by applying various transformations.
    
    This is a convenience function that uses the singleton parser instance.
    
    Args:
        formula: Formula string or AST node
        
    Returns:
        An optimized version of the AST
        
    Examples:
        >>> ast = optimize_formula("2 + 3")
        >>> isinstance(ast, Literal)
        True
        >>> ast.value
        5.0
    """
    if isinstance(formula, str):
        node = parse_formula(formula)
    else:
        node = formula
    
    return _parser.optimize_ast(node)

def validate_formula(formula: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that a formula conforms to the defined grammar.
    
    This is a convenience function that uses the singleton parser instance.
    
    Args:
        formula: The formula string to validate
        
    Returns:
        A tuple of (is_valid, error_message) where is_valid is a boolean and
        error_message is None if valid or a string describing the error
        
    Examples:
        >>> validate_formula("x + y * z")
        (True, None)
        >>> validate_formula("x +* y")
        (False, 'Invalid formula syntax...')
    """
    return _parser.validate_against_grammar(formula) 