# Hindsight Library Tests

This directory contains unit tests for the hindsight library.

## Directory Structure

The tests are organized to mirror the structure of the main library:

```
tests/
├── data/
│   ├── ast/           # Tests for the AST module
│   │   ├── test_ast_core.py
│   │   ├── test_function_registry.py
│   │   └── ...
│   └── ...
└── ...
```

## Running Tests

### Run All Tests

To run all tests:

```bash
python tests/run_tests.py
```

### Run Tests for a Specific Module

To run tests for a specific module:

```bash
python tests/run_tests.py --module data/ast
```

### Verbose Output

For more detailed test output:

```bash
python tests/run_tests.py --verbose
```

## Writing Tests

When adding new tests, follow these guidelines:

1. Create test files with names that start with `test_`.
2. Place test files in the directory corresponding to the module being tested.
3. Use the `unittest` framework for test cases.
4. Include detailed docstrings and comments to explain what is being tested.
5. Use meaningful test method names that describe what is being tested.

Example:

```python
import unittest

class TestMyFeature(unittest.TestCase):
    """Tests for the MyFeature functionality."""

    def setUp(self):
        """Set up test environment."""
        # Set up code here

    def test_feature_works_with_normal_input(self):
        """Test that the feature works with normal input."""
        # Test code here
        self.assertEqual(expected, actual)

    def tearDown(self):
        """Clean up after tests."""
        # Clean up code here
```

## Code Coverage

To generate code coverage reports, install the `coverage` package and run:

```bash
coverage run --source=src tests/run_tests.py
coverage report
```

For HTML reports:

```bash
coverage html
```

Then open `htmlcov/index.html` in your browser. 