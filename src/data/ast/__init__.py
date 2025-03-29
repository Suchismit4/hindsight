"""
Abstract Syntax Tree (AST) module for financial formula parsing and execution.

This module provides functionality for parsing financial formulas using a context-free
grammar (CFG) and executing them on financial data. It is designed to be compatible
with JAX's JIT compilation for efficient computation.
"""

from .nodes import (
    Node, 
    BinaryOp, 
    UnaryOp, 
    Literal, 
    Variable, 
    FunctionCall
)

from .parser import parse_formula, extract_variables, extract_functions, optimize_formula
from .functions import register_function, get_registered_functions, get_function_context
from .grammar import get_grammar_description, FINANCIAL_FORMULA_GRAMMAR
from .visualization import (
    visualize_ast, 
    visualize_parse_tree, 
    visualize_jit_graph, 
    compare_asts, 
    visualize_formula_comparison
)

__all__ = [
    # Core node classes
    'Node',
    'BinaryOp', 
    'UnaryOp', 
    'Literal', 
    'Variable', 
    'FunctionCall',
    
    # Parser functions
    'parse_formula',
    'extract_variables',
    'extract_functions',
    'optimize_formula',
    
    # Function registry
    'register_function',
    'get_registered_functions',
    'get_function_context',
    
    # Grammar definition
    'get_grammar_description',
    'FINANCIAL_FORMULA_GRAMMAR',
    
    # Visualization
    'visualize_ast',
    'visualize_parse_tree',
    'visualize_jit_graph',
    'compare_asts',
    'visualize_formula_comparison'
] 