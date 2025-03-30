"""
AST node classes for financial formula parsing.

This module defines the node classes used to represent the abstract syntax tree (AST)
for financial formulas. These classes are implemented using Equinox for compatibility
with JAX's PyTree and JIT functionality.

The nodes are designed to be immutable, serializable, and compatible with JAX transformations
such as jit, vmap, and grad. This allows for efficient evaluation of financial formulas
on large datasets with automatic differentiation support.

Examples:
    Creating and evaluating a simple expression:
    >>> from src.data.ast.nodes import BinaryOp, Literal, Variable
    >>> expr = BinaryOp('+', Literal(2.0), BinaryOp('*', Variable('x'), Literal(3.0)))
    >>> expr.evaluate({'x': 4.0})
    14.0
    
    Using with JAX transformations:
    >>> import jax
    >>> def f(x):
    ...     context = {'x': x}
    ...     return expr.evaluate(context)
    >>> jit_f = jax.jit(f)
    >>> jit_f(4.0)
    Array(14., dtype=float32)
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Set
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import xarray as xr
import xarray_jax

# Abstract base class for all AST nodes
class Node(eqx.Module):
    """
    Base class for all AST nodes.
    
    This abstract class defines the interface that all AST nodes must implement.
    It ensures compatibility with JAX's PyTree and JIT functionality through Equinox.
    
    All nodes must implement the evaluate method, which computes the result of
    the node given a context of variable values.
    """
    
    def evaluate(self, context: Dict[str, Any]) -> Union[jnp.ndarray, xr.DataArray, xr.Dataset]:
        """
        Evaluate the node with the given context.
        
        Args:
            context: Dictionary mapping variable names to their values and
                    function names to their implementations. Function names
                    in the context should be prefixed with "_func_".
            
        Returns:
            The result of evaluating the node as a JAX ndarray, xarray DataArray, or xarray Dataset
            
        Raises:
            NotImplementedError: If the subclass does not implement this method
            ValueError: If a variable or function is missing from the context
        """
        raise NotImplementedError("Subclasses must implement evaluate")
    
    def get_variables(self) -> Set[str]:
        """
        Get all variable names used in this node and its children.
        
        Returns:
            A set of variable names
            
        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement get_variables")
    
    def get_functions(self) -> Set[str]:
        """
        Get all function names used in this node and its children.
        
        Returns:
            A set of function names
            
        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement get_functions")
    
    def __str__(self) -> str:
        """
        Convert the node to a string representation.
        
        Returns:
            A string representation of the node
            
        Raises:
            NotImplementedError: If the subclass does not implement this method
        """
        raise NotImplementedError("Subclasses must implement __str__")

# Node for binary operations (e.g., +, -, *, /)
class BinaryOp(Node):
    """
    Node representing a binary operation.
    
    This node represents a binary operation between two operands, such as
    addition, subtraction, multiplication, division, or exponentiation.
    
    Attributes:
        op: The operation to perform (e.g., '+', '-', '*', '/', '^')
        left: The left operand
        right: The right operand
        
    Examples:
        >>> BinaryOp('+', Literal(2.0), Literal(3.0)).evaluate({})
        Array(5., dtype=float32)
        >>> BinaryOp('*', Variable('x'), Literal(3.0)).evaluate({'x': 4.0})
        Array(12., dtype=float32)
    """
    op: str
    left: Node
    right: Node
    
    def evaluate(self, context: Dict[str, Any]) -> Union[jnp.ndarray, xr.DataArray, xr.Dataset]:
        """
        Evaluate the binary operation.
        
        Args:
            context: Dictionary mapping variable names to their values
            
        Returns:
            The result of the binary operation
            
        Raises:
            ValueError: If the operator is unknown
        """
        left_val = self.left.evaluate(context)
        right_val = self.right.evaluate(context)
        
        # Check if we're dealing with xarray objects
        is_xarray = isinstance(left_val, (xr.DataArray, xr.Dataset)) or isinstance(right_val, (xr.DataArray, xr.Dataset))
        
        if self.op == '+':
            return left_val + right_val
        elif self.op == '-':
            return left_val - right_val
        elif self.op == '*':
            return left_val * right_val
        elif self.op == '/':
            if is_xarray:
                # For xarray objects, use the divide method to handle division by zero
                return left_val / right_val
            else:
                # For JAX arrays, handle division by zero gracefully by returning NaN
                return jnp.where(right_val == 0.0, jnp.nan, left_val / right_val)
        elif self.op == '^':
            if is_xarray:
                # Use xarray's implementation for power
                return left_val ** right_val
            else:
                return jnp.power(left_val, right_val)
        else:
            # This should never happen if the parser is correct
            raise ValueError(f"Unknown binary operator: {self.op}")
    
    def get_variables(self) -> Set[str]:
        """
        Get all variable names used in this node and its children.
        
        Returns:
            A set of variable names
        """
        return self.left.get_variables() | self.right.get_variables()
    
    def get_functions(self) -> Set[str]:
        """
        Get all function names used in this node and its children.
        
        Returns:
            A set of function names
        """
        return self.left.get_functions() | self.right.get_functions()
    
    def __str__(self) -> str:
        """
        Convert the binary operation to a string representation.
        
        Returns:
            A string representation of the binary operation
        """
        left_str = str(self.left)
        right_str = str(self.right)
        
        # Add parentheses around the operands if they are also binary operations
        # with lower precedence to ensure correct precedence in the string representation
        if isinstance(self.left, BinaryOp) and self._has_lower_precedence(self.left.op, self.op):
            left_str = f"({left_str})"
            
        if isinstance(self.right, BinaryOp) and (
            self._has_lower_precedence(self.right.op, self.op) or 
            (self.op in {'-', '/'} and self.right.op == self.op)  # Right associative for - and /
        ):
            right_str = f"({right_str})"
            
        return f"{left_str} {self.op} {right_str}"
    
    @staticmethod
    def _has_lower_precedence(op1: str, op2: str) -> bool:
        """
        Check if op1 has lower precedence than op2.
        
        Args:
            op1: The first operator
            op2: The second operator
            
        Returns:
            True if op1 has lower precedence than op2
        """
        precedence = {
            '+': 1, '-': 1,
            '*': 2, '/': 2,
            '^': 3
        }
        return precedence.get(op1, 0) < precedence.get(op2, 0)

# Node for unary operations (e.g., -)
class UnaryOp(Node):
    """
    Node representing a unary operation.
    
    This node represents a unary operation on an operand, such as
    negation (unary minus).
    
    Attributes:
        op: The operation to perform (e.g., '-')
        operand: The operand
        
    Examples:
        >>> UnaryOp('-', Literal(3.0)).evaluate({})
        Array(-3., dtype=float32)
    """
    op: str
    operand: Node
    
    def evaluate(self, context: Dict[str, Any]) -> jnp.ndarray:
        """
        Evaluate the unary operation.
        
        Args:
            context: Dictionary mapping variable names to their values
            
        Returns:
            The result of the unary operation
            
        Raises:
            ValueError: If the operator is unknown
        """
        val = self.operand.evaluate(context)
        
        if self.op == '-':
            return -val
        else:
            # This should never happen if the parser is correct
            raise ValueError(f"Unknown unary operator: {self.op}")
    
    def get_variables(self) -> Set[str]:
        """
        Get all variable names used in this node and its children.
        
        Returns:
            A set of variable names
        """
        return self.operand.get_variables()
    
    def get_functions(self) -> Set[str]:
        """
        Get all function names used in this node and its children.
        
        Returns:
            A set of function names
        """
        return self.operand.get_functions()
    
    def __str__(self) -> str:
        """
        Convert the unary operation to a string representation.
        
        Returns:
            A string representation of the unary operation
        """
        operand_str = str(self.operand)
        
        # Add parentheses around the operand if it is a binary operation
        # to ensure correct precedence in the string representation
        if isinstance(self.operand, BinaryOp):
            operand_str = f"({operand_str})"
            
        return f"{self.op}{operand_str}"

# Node for literals (numbers)
class Literal(Node):
    """
    Node representing a literal value.
    
    This node represents a constant numeric value in the AST.
    
    Attributes:
        value: The literal value as a float
        
    Examples:
        >>> Literal(3.14).evaluate({})
        Array(3.14, dtype=float32)
    """
    value: float
    
    def evaluate(self, context: Dict[str, Any]) -> jnp.ndarray:
        """
        Evaluate the literal value.
        
        Args:
            context: Dictionary mapping variable names to their values (not used)
            
        Returns:
            The literal value as a JAX array
        """
        return jnp.array(self.value)
    
    def get_variables(self) -> Set[str]:
        """
        Get all variable names used in this node and its children.
        
        Returns:
            An empty set since literals don't use variables
        """
        return set()
    
    def get_functions(self) -> Set[str]:
        """
        Get all function names used in this node and its children.
        
        Returns:
            An empty set since literals don't use functions
        """
        return set()
    
    def __str__(self) -> str:
        """
        Convert the literal to a string representation.
        
        Returns:
            A string representation of the literal value
        """
        # Format the value as an integer if it's a whole number
        if self.value == int(self.value):
            return str(int(self.value))
        return str(self.value)

# Node for variables
class Variable(Node):
    """
    Node representing a variable.
    
    This node represents a variable that will be looked up in the evaluation context.
    
    Attributes:
        name: The name of the variable
        
    Examples:
        >>> Variable('x').evaluate({'x': 5.0})
        Array(5., dtype=float32)
    """
    name: str
    
    def evaluate(self, context: Dict[str, Any]) -> Union[jnp.ndarray, xr.DataArray, xr.Dataset]:
        """
        Evaluate the variable by looking up its value in the context.
        
        Args:
            context: Dictionary mapping variable names to their values
            
        Returns:
            The value of the variable
            
        Raises:
            ValueError: If the variable is not in the context
        """
        if self.name not in context:
            raise ValueError(f"Variable '{self.name}' not found in context")
        
        return context[self.name]
    
    def get_variables(self) -> Set[str]:
        """
        Get all variable names used in this node and its children.
        
        Returns:
            A set containing the variable name
        """
        return {self.name}
    
    def get_functions(self) -> Set[str]:
        """
        Get all function names used in this node and its children.
        
        Returns:
            An empty set since variables don't use functions
        """
        return set()
    
    def __str__(self) -> str:
        """
        Convert the variable to a string representation.
        
        Returns:
            The variable name
        """
        return self.name

# Node for function calls
class FunctionCall(Node):
    """
    Node representing a function call.
    
    This node represents a call to a function with arguments.
    
    Attributes:
        name: The name of the function
        args: The argument nodes
        
    Examples:
        >>> func = lambda a, b: a + b
        >>> FunctionCall('add', [Literal(2.0), Literal(3.0)]).evaluate({'_func_add': func})
        Array(5., dtype=float32)
    """
    name: str
    args: List[Node]
    
    def evaluate(self, context: Dict[str, Any]) -> Union[jnp.ndarray, xr.DataArray, xr.Dataset]:
        """
        Evaluate the function call by evaluating the arguments and calling the function.
        Passes the value of Literal nodes directly as Python scalars.
        
        Args:
            context: Dictionary mapping variable names to their values and
                    function names to their implementations
            
        Returns:
            The result of the function call
            
        Raises:
            ValueError: If the function is not in the context
        """
        func_key = f"_func_{self.name}"
        if func_key not in context:
            raise ValueError(f"Function '{self.name}' not found in context")
        
        # Get the function from the context
        func = context[func_key]
        
        # Evaluate arguments, but pass Literal values directly
        processed_args = []
        for arg_node in self.args:  # Iterate through the argument *nodes*
            if isinstance(arg_node, Literal):
                # If it's a Literal node, use its Python value directly.
                # Ensure it's int if it represents an integer.
                val = arg_node.value
                if val == int(val):
                    processed_args.append(int(val))
                else:
                    processed_args.append(val)
            else:
                # Otherwise, evaluate the node.
                evaluated_arg = arg_node.evaluate(context)
                processed_args.append(evaluated_arg)

        # Call the function with the processed arguments
        try:
            return func(*processed_args)
        except Exception as e:
            arg_types = [type(a) for a in processed_args]
            print(f"Error calling {self.name} with arg types: {arg_types}")
            raise ValueError(f"Error calling function '{self.name}': {str(e)}")
    
    def get_variables(self) -> Set[str]:
        """
        Get all variable names used in this node and its children.
        
        Returns:
            A set of variable names from all arguments
        """
        variables = set()
        for arg in self.args:
            variables |= arg.get_variables()
        return variables
    
    def get_functions(self) -> Set[str]:
        """
        Get all function names used in this node and its children.
        
        Returns:
            A set containing this function name and all function names from arguments
        """
        functions = {self.name}
        for arg in self.args:
            functions |= arg.get_functions()
        return functions
    
    def __str__(self) -> str:
        """
        Convert the function call to a string representation.
        
        Returns:
            A string representation of the function call
        """
        arg_strs = [str(arg) for arg in self.args]
        return f"{self.name}({', '.join(arg_strs)})" 