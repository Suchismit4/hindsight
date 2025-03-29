"""
Visualization utilities for AST nodes.

This module provides comprehensive visualization tools for Abstract Syntax Trees (ASTs)
to aid in debugging, documentation, and educational purposes. It can generate graphical
representations of AST structures, parse trees, and JAX computation graphs.

The visualizations are created using the Graphviz library and can be output in various
formats including PNG, SVG, PDF, and more. The module supports customization of node
colors, layout algorithms, and other visual attributes.

Key features:
- Visualize individual AST nodes and their relationships
- Generate parse tree visualizations from formula strings
- Visualize JAX computation graphs after JIT compilation
- Compare multiple ASTs side by side
- Customize colors, styles, and layout of visualizations

Examples:
    >>> from src.data.ast.parser import parse_formula
    >>> from src.data.ast.visualization import visualize_ast
    >>> formula = "x + y * z"
    >>> ast = parse_formula(formula)
    >>> output_file = visualize_ast(ast, output_path="visualization", view=True)
    >>> print(f"Visualization saved to: {output_file}")

Note:
    This module requires the Graphviz library to be installed and accessible.
    Both the Python package (pip install graphviz) and the system-level Graphviz
    binaries must be installed.
"""

import os
import graphviz
import tempfile
from typing import Optional, Dict, Any, Union, List, Tuple
import matplotlib.pyplot as plt
from PIL import Image

from .nodes import Node, BinaryOp, UnaryOp, Literal, Variable, FunctionCall

# Default color scheme for different node types
DEFAULT_NODE_COLORS = {
    "BinaryOp": "#FFCCCC",  # Light red
    "UnaryOp": "#CCFFCC",   # Light green
    "Literal": "#CCCCFF",   # Light blue
    "Variable": "#FFFFCC",  # Light yellow
    "FunctionCall": "#FFCCFF"  # Light purple
}

def visualize_ast(
    node: Node, 
    output_path: str = "ast_visualization", 
    view: bool = True,
    format: str = "png",
    node_colors: Optional[Dict[str, str]] = None,
    layout: str = "dot",
    dpi: int = 300,
    show_types: bool = False,
    title: Optional[str] = None
) -> str:
    """
    Visualize an AST tree and save it as an image.
    
    This function creates a graphical representation of an Abstract Syntax Tree (AST)
    and saves it to disk. The visualization shows the structure of the AST with
    nodes colored according to their type and edges connecting parent nodes to
    their children.
    
    Args:
        node: The root AST node to visualize
        output_path: The path where the visualization should be saved (without extension)
        view: Whether to open the visualization after creation
        format: The output format (png, pdf, svg, etc.)
        node_colors: Custom colors for different node types. If None, default colors are used.
        layout: The Graphviz layout algorithm to use (dot, neato, fdp, sfdp, twopi, circo)
        dpi: The resolution of the output image (dots per inch)
        show_types: Whether to show the type names of nodes
        title: Optional title to display at the top of the visualization
        
    Returns:
        The path to the saved visualization file
        
    Raises:
        ValueError: If the layout algorithm is not supported
        OSError: If Graphviz is not installed or fails to generate the visualization
        
    Examples:
        >>> from src.data.ast.parser import parse_formula
        >>> formula = "x + y * z"
        >>> ast = parse_formula(formula)
        >>> output_file = visualize_ast(
        ...     ast,
        ...     output_path="my_visualization",
        ...     view=False,
        ...     format="png",
        ...     node_colors={"BinaryOp": "#FF0000"}
        ... )
        >>> print(f"Visualization saved to: {output_file}")
    """
    # Validate layout algorithm
    valid_layouts = ["dot", "neato", "fdp", "sfdp", "twopi", "circo"]
    if layout not in valid_layouts:
        raise ValueError(f"Invalid layout algorithm: {layout}. " 
                        f"Supported layouts are: {', '.join(valid_layouts)}")
    
    # Use default colors if none provided
    if node_colors is None:
        node_colors = DEFAULT_NODE_COLORS.copy()
    
    # Create a new directed graph with specified layout
    dot = graphviz.Digraph(
        comment='AST Visualization',
        engine=layout,
        format=format,
        graph_attr={
            'dpi': str(dpi),
            'rankdir': 'TB',  # Top to bottom layout
            'fontname': 'Helvetica',
            'bgcolor': 'white'
        },
        node_attr={
            'fontname': 'Helvetica',
            'shape': 'box',
            'style': 'filled',
            'margin': '0.1,0.1'
        },
        edge_attr={
            'fontname': 'Helvetica',
            'fontsize': '10'
        }
    )
    
    # Add title if provided
    if title:
        dot.attr(label=title)
        dot.attr(labelloc='t')  # Place label at top
        dot.attr(labeljust='c')  # Center label
    
    # Counter for unique node IDs
    counter = [0]
    
    def add_node_to_graph(node: Node, parent_id: Optional[str] = None, edge_label: Optional[str] = None) -> str:
        """
        Recursively add nodes to the graph.
        
        Args:
            node: Node to add
            parent_id: ID of the parent node (if any)
            edge_label: Label for the edge from parent to this node (if any)
            
        Returns:
            ID of the added node
        """
        node_id = f"node_{counter[0]}"
        counter[0] += 1
        
        node_type = node.__class__.__name__
        color = node_colors.get(node_type, "#FFFFFF")  # Default to white if not specified
        
        # Create different labels based on node type
        if isinstance(node, BinaryOp):
            type_prefix = f"{node_type}\\n" if show_types else ""
            label = f"{type_prefix}Operator: {node.op}"
        elif isinstance(node, UnaryOp):
            type_prefix = f"{node_type}\\n" if show_types else ""
            label = f"{type_prefix}Operator: {node.op}"
        elif isinstance(node, Literal):
            type_prefix = f"{node_type}\\n" if show_types else ""
            label = f"{type_prefix}Value: {node.value}"
        elif isinstance(node, Variable):
            type_prefix = f"{node_type}\\n" if show_types else ""
            label = f"{type_prefix}Name: {node.name}"
        elif isinstance(node, FunctionCall):
            type_prefix = f"{node_type}\\n" if show_types else ""
            label = f"{type_prefix}Function: {node.name}"
        else:
            label = node_type
        
        # Add the node to the graph
        dot.node(node_id, label, fillcolor=color)
        
        # Connect to parent if exists with edge label if provided
        if parent_id and edge_label:
            dot.edge(parent_id, node_id, label=edge_label)
        elif parent_id:
            dot.edge(parent_id, node_id)
        
        # Recursively add children
        if isinstance(node, BinaryOp):
            # Add left and right children with edge labels
            left_id = add_node_to_graph(node.left, node_id, "left")
            right_id = add_node_to_graph(node.right, node_id, "right")
        elif isinstance(node, UnaryOp):
            # Add operand child with edge label
            operand_id = add_node_to_graph(node.operand, node_id, "operand")
        elif isinstance(node, FunctionCall):
            # Add argument children with edge labels
            for i, arg in enumerate(node.args):
                arg_id = add_node_to_graph(arg, node_id, f"arg{i}")
        
        return node_id
    
    # Start the recursive process - no parent for the root
    add_node_to_graph(node)
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    try:
        # Save the visualization
        output_file = dot.render(filename=output_path, format=format, cleanup=True, view=view)
        return output_file
    except Exception as e:
        raise OSError(f"Failed to generate visualization with Graphviz: {str(e)}. "
                      f"Ensure Graphviz is installed on your system.")

def visualize_parse_tree(
    formula: str,
    output_path: str = "parse_tree_visualization",
    view: bool = True,
    format: str = "png",
    layout: str = "dot",
    dpi: int = 300,
    show_types: bool = False,
    sanitize_filename: bool = True
) -> str:
    """
    Visualize the parse tree for a formula and save it as an image.
    
    This is a convenience function that parses the formula and then
    visualizes its AST. It automatically sanitizes the output filename
    to avoid issues with special characters.
    
    Args:
        formula: The formula string to visualize
        output_path: The path where the visualization should be saved (without extension)
        view: Whether to open the visualization after creation
        format: The output format (png, pdf, svg, etc.)
        layout: The Graphviz layout algorithm to use (dot, neato, fdp, sfdp, twopi, circo)
        dpi: The resolution of the output image (dots per inch)
        show_types: Whether to show the type names of nodes
        sanitize_filename: Whether to sanitize the filename to avoid special characters
        
    Returns:
        The path to the saved visualization file
        
    Examples:
        >>> output_file = visualize_parse_tree(
        ...     "x + y * z",
        ...     output_path="parse_tree",
        ...     view=False
        ... )
        >>> print(f"Parse tree visualization saved to: {output_file}")
    """
    from .parser import parse_formula
    
    # Parse the formula to get the AST
    ast = parse_formula(formula)
    
    # Add formula as a title for the visualization
    title = f"Formula: {formula}"
    
    # Create a safe filename by replacing problematic characters
    if sanitize_filename:
        safe_formula = formula.replace(' ', '_')
        for char in '()*/\\^:;<>?|"\'':
            safe_formula = safe_formula.replace(char, '')
        
        # Generate the output path
        output_with_formula = f"{output_path}_{safe_formula}"
    else:
        output_with_formula = f"{output_path}_{formula.replace(' ', '_')}"
    
    # Ensure the output path isn't too long by truncating if necessary
    if len(output_with_formula) > 100:
        output_with_formula = output_with_formula[:100]
    
    # Visualize the AST
    return visualize_ast(
        ast, 
        output_path=output_with_formula,
        view=view, 
        format=format,
        layout=layout,
        dpi=dpi,
        show_types=show_types,
        title=title
    )

def visualize_jit_graph(
    function: Any,
    args: Any,
    output_path: str = "jit_graph_visualization",
    view: bool = True,
    format: str = "png",
    dpi: int = 300
) -> str:
    """
    Visualize the JAX computation graph after JIT compilation.
    
    This uses JAX's lower().compiler_ir() feature to get the HLO representation
    of the compiled function. Note that this is a higher level visualization
    and does not show the full detail of the JAX computation graph.
    
    Args:
        function: A JAX-compatible function (not yet JIT-compiled)
        args: Arguments to pass to the function
        output_path: The path where the visualization should be saved (without extension)
        view: Whether to open the visualization after creation
        format: The output format (png, pdf, svg, etc.)
        dpi: The resolution of the output image (dots per inch)
        
    Returns:
        The path to the saved visualization file
        
    Raises:
        ImportError: If JAX is not installed
        OSError: If Graphviz is not installed or fails to generate the visualization
        
    Examples:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def my_function(x, y):
        ...     return x + y * jnp.sin(x)
        >>> output_file = visualize_jit_graph(
        ...     my_function,
        ...     (jnp.array(1.0), jnp.array(2.0)),
        ...     output_path="jit_graph",
        ...     view=False
        ... )
        >>> print(f"JIT graph visualization saved to: {output_file}")
    """
    import jax
    
    try:
        # Create a jitted version of the function if it's not already jitted
        if not hasattr(function, 'lower'):
            jitted_func = jax.jit(function)
        else:
            jitted_func = function
        
        # Lower the function to get the compiler IR
        lowered = jitted_func.lower(*args)
        hlo_text = lowered.compiler_ir('hlo').as_hlo_text()
        
        # Create a new directed graph
        dot = graphviz.Digraph(
            comment='JAX Computation Graph',
            format=format,
            graph_attr={
                'dpi': str(dpi),
                'rankdir': 'TB',
                'fontname': 'Helvetica',
                'bgcolor': 'white'
            },
            node_attr={
                'fontname': 'Helvetica',
                'shape': 'box',
                'style': 'filled',
                'fillcolor': '#CCFFFF'
            }
        )
        
        # Add a single node with the HLO text summary
        dot.node('hlo', label=f"HLO Computation\n(See .hlo file for details)", shape='box')
        
        # Add a note about the function
        func_name = function.__name__ if hasattr(function, '__name__') else "anonymous function"
        dot.node('info', label=f"Function: {func_name}\nArgs: {args}", shape='note', fillcolor='#FFFFCC')
        
        # Save the HLO text to a separate file
        hlo_file = f"{output_path}.hlo"
        with open(hlo_file, 'w') as f:
            f.write(hlo_text)
        
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the visualization
        output_file = dot.render(filename=output_path, format=format, cleanup=True, view=view)
        
        print(f"HLO text saved to {hlo_file}")
        return output_file
        
    except Exception as e:
        print(f"Warning: Couldn't generate HLO: {str(e)}")
        print("Try using a simple non-jitted function.")
        
        # Create a simple diagram indicating the error
        dot = graphviz.Digraph(comment='JAX Computation Graph')
        dot.node('error', label=f"Couldn't visualize JIT graph.\nError: {str(e)}", color='red')
        
        # Save the visualization
        output_file = dot.render(filename=output_path, format=format, cleanup=True, view=view)
        return output_file

def compare_asts(
    nodes: List[Tuple[Node, str]],
    output_path: str = "ast_comparison",
    view: bool = True,
    format: str = "png",
    node_colors: Optional[Dict[str, str]] = None,
    layout: str = "dot",
    dpi: int = 300,
    show_types: bool = False,
    horizontal: bool = True
) -> str:
    """
    Compare multiple ASTs side by side.
    
    This function creates a visualization with multiple ASTs arranged
    horizontally or vertically for comparison.
    
    Args:
        nodes: List of (node, label) tuples to compare
        output_path: The path where the visualization should be saved (without extension)
        view: Whether to open the visualization after creation
        format: The output format (png, pdf, svg, etc.)
        node_colors: Custom colors for different node types
        layout: The Graphviz layout algorithm to use
        dpi: The resolution of the output image (dots per inch)
        show_types: Whether to show the type names of nodes
        horizontal: Whether to arrange the ASTs horizontally (True) or vertically (False)
        
    Returns:
        The path to the saved visualization file
        
    Examples:
        >>> from src.data.ast.parser import parse_formula
        >>> ast1 = parse_formula("x + y")
        >>> ast2 = parse_formula("x * y")
        >>> output_file = compare_asts(
        ...     [(ast1, "Addition"), (ast2, "Multiplication")],
        ...     output_path="comparison",
        ...     view=False
        ... )
        >>> print(f"Comparison visualization saved to: {output_file}")
    """
    # Use temporary files for individual visualizations
    temp_files = []
    
    for node, label in nodes:
        # Create a temporary file
        fd, temp_file = tempfile.mkstemp(suffix=f".{format}")
        os.close(fd)
        
        # Generate visualization for this node
        visualize_ast(
            node,
            output_path=temp_file.replace(f".{format}", ""),
            view=False,
            format=format,
            node_colors=node_colors,
            layout=layout,
            dpi=dpi,
            show_types=show_types,
            title=label
        )
        
        # Add the temporary file to the list
        temp_files.append(temp_file)
    
    # Create the comparison visualization
    if format.lower() in ['png', 'jpg', 'jpeg']:
        # For image formats, use PIL to combine images
        images = [Image.open(file) for file in temp_files]
        
        # Calculate dimensions for the combined image
        if horizontal:
            width = sum(img.width for img in images)
            height = max(img.height for img in images)
        else:
            width = max(img.width for img in images)
            height = sum(img.height for img in images)
        
        # Create a new image with the combined dimensions
        combined = Image.new('RGB', (width, height), (255, 255, 255))
        
        # Paste each image in the right position
        x_offset = 0
        y_offset = 0
        for img in images:
            if horizontal:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            else:
                combined.paste(img, (0, y_offset))
                y_offset += img.height
        
        # Save the combined image
        combined_path = f"{output_path}.{format}"
        combined.save(combined_path, dpi=(dpi, dpi))
        
        # Remove temporary files
        for file in temp_files:
            try:
                os.remove(file)
            except:
                pass
        
        # Open the combined image if requested
        if view:
            try:
                import webbrowser
                webbrowser.open(combined_path)
            except:
                pass
        
        return combined_path
    else:
        # For vector formats like SVG, PDF, etc., create a multi-page document
        # This is more complex and would require additional libraries
        # For now, just return the individual files
        print(f"Multiple AST visualizations saved to: {', '.join(temp_files)}")
        return temp_files[0]

def visualize_formula_comparison(
    formulas: List[str],
    output_path: str = "formula_comparison",
    view: bool = True,
    format: str = "png",
    optimize: bool = False,
    horizontal: bool = True
) -> str:
    """
    Compare multiple formulas by visualizing their parse trees side by side.
    
    This function parses multiple formulas and displays their ASTs for comparison.
    
    Args:
        formulas: List of formula strings to compare
        output_path: The path where the visualization should be saved (without extension)
        view: Whether to open the visualization after creation
        format: The output format (png, pdf, svg, etc.)
        optimize: Whether to optimize the ASTs before visualization
        horizontal: Whether to arrange the ASTs horizontally (True) or vertically (False)
        
    Returns:
        The path to the saved visualization file
        
    Examples:
        >>> output_file = visualize_formula_comparison(
        ...     ["x + y", "x * y", "x / (y + 1)"],
        ...     output_path="formula_comparison",
        ...     view=False
        ... )
        >>> print(f"Formula comparison saved to: {output_file}")
    """
    from .parser import parse_formula, optimize_formula
    
    # Parse and optionally optimize the formulas
    nodes = []
    for formula in formulas:
        ast = parse_formula(formula)
        if optimize:
            ast = optimize_formula(ast)
        nodes.append((ast, formula))
    
    # Compare the ASTs
    return compare_asts(
        nodes,
        output_path=output_path,
        view=view,
        format=format,
        horizontal=horizontal
    ) 