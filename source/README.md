# Hindsight Pipeline Framework Documentation

This directory contains the Sphinx documentation for the Hindsight Pipeline Framework.

## Building the Documentation

To build the HTML documentation:

1. Activate the conda environment with Sphinx installed:
   ```bash
   conda activate jax
   ```

2. Build the documentation:
   ```bash
   make html
   ```

3. Serve the documentation locally (recommended for VM/remote development):
   ```bash
   python3 serve_docs.py
   ```
   This will start a local HTTP server on port 8000. VS Code will automatically port-forward the URL for viewing in your browser.

   Alternatively, you can open the documentation directly:
   ```bash
   python3 open_docs.py
   ```

The generated documentation will be available in `build/html/index.html`.

## Documentation Structure

### Getting Started Guide
- **Overview**: Framework architecture and core principles
- **Data Loading**: Using DataManager to load financial datasets
- **Data Handler**: Configuring and using the data processing pipeline
- **Feature Engineering**: Working with processors and formula evaluation
- **Walk-Forward Analysis**: Temporal segmentation and robust backtesting
- **Model Integration**: Integrating ML models with the pipeline
- **Execution and Analysis**: Running complete workflows and analyzing results

### API Reference
- **Pipeline API**: Main entry point and core classes
- **Data Handler API**: Data processing pipeline components
- **Walk-Forward API**: Temporal segmentation and execution
- **Model API**: Model integration and adapters

### Examples
- **Complete Workflow**: End-to-end example following `example.py`

## Key Features Highlighted

The documentation focuses on the high-level abstractions and public API that users need to understand:

- **Separation of "How" and "When"**: Clear distinction between data processing and temporal logic
- **Three-Stage Processing**: Shared, learn, and infer processor stages
- **Temporal Validity**: Prevention of lookahead bias through proper state management
- **Walk-Forward Analysis**: Robust backtesting with configurable temporal segments
- **Model Integration**: Seamless integration of ML models with the pipeline

## Areas Not Covered (Marked as TBA)

The documentation intentionally avoids implementation details that are "under the hood":
- AST system internals and formula evaluation implementation
- Data module internals (loaders, processors, core operations)
- Backtester implementation details
- Rolling operations and masking specifics

These topics may be covered in future detailed developer documentation.
