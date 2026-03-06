# 📚 Hindsight Pipeline Framework Documentation

## Quick Start

### 1. Build the Documentation
```bash
conda activate jax
make html
```

### 2. View the Documentation
```bash
python3 serve_docs.py
```

This will:
- Start a local HTTP server on `http://localhost:8000`
- VS Code will automatically detect and port-forward the URL
- Click the port-forwarded link in VS Code to view the documentation in your browser

### 3. Stop the Server
Press `Ctrl+C` in the terminal running the server.

## What's Included

✅ **Complete Getting Started Guide** (7 detailed pages):
- Framework Overview & Architecture
- Data Loading with DataManager
- Data Handler & Processing Pipeline
- Feature Engineering & Processors
- Walk-Forward Analysis & Temporal Segmentation
- Model Integration & Adapters
- Pipeline Execution & Results Analysis

✅ **Comprehensive API Reference**:
- Pipeline API (main entry point)
- Data Handler API (processors, configuration)
- Walk-Forward API (segments, planning, execution)
- Model API (adapters, runners)

✅ **Complete Workflow Example**:
- Step-by-step example following `example.py`
- Code samples with detailed explanations

## Key Features Documented

- **Separation of "How" vs "When"**: Data processing vs temporal logic
- **Three-Stage Processing**: Shared, learn, infer processors
- **Temporal Validity**: Prevention of lookahead bias
- **Walk-Forward Analysis**: Robust backtesting methodology
- **Model Integration**: Seamless ML model integration
- **Public API**: All user-facing functions and classes

The documentation focuses on high-level abstractions and practical usage while keeping implementation details as "under the hood" topics for future detailed developer docs.

---

🚀 **Ready to explore your framework's capabilities!**
