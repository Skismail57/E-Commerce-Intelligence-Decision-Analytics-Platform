# Contributing to E-Commerce Intelligence Platform

Thank you for your interest in contributing to the E-Commerce Intelligence & Decision Analytics Platform! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL 16+
- Git
- Docker & Docker Compose (optional)

### Setting Up Development Environment

1. **Fork the repository**
   ```bash
   # Fork the repository on GitHub
   # Clone your fork
   git clone https://github.com/your-username/ecommerce-intelligence.git
   cd ecommerce-intelligence
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pytest pytest-cov black flake8
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

5. **Run tests**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Emergency fixes

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write code**
   - Follow PEP 8 style guidelines
   - Add docstrings to functions and classes
   - Write tests for new functionality

2. **Run tests**
   ```bash
   pytest tests/ -v --cov=src
   ```

3. **Format code**
   ```bash
   black src/ tests/
   ```

4. **Lint code**
   ```bash
   flake8 src/ tests/
   ```

### Committing Changes

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(analytics): add customer churn prediction model

Implement XGBoost-based churn prediction with:
- Feature engineering for customer behavior
- Model training and evaluation
- Risk tier classification (Low/Medium/High)

Closes #123
```

### Submitting a Pull Request

1. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Go to GitHub and create a PR from your branch to `develop`
   - Fill in the PR template
   - Link related issues

3. **PR Review Process**
   - Automated tests will run
   - Code review by maintainers
   - Address feedback
   - Merge when approved

## Project Structure Guidelines

### Adding New Analytics

1. **Create module in `src/analytics/`**
   ```python
   # src/analytics/new_analytics.py
   from pathlib import Path
   from typing import Dict, Optional
   import pandas as pd
   
   class NewAnalytics:
       def __init__(self, data_dir: Optional[Path] = None):
           self.data_dir = Path(data_dir) if data_dir else None
       
       def analyze(self) -> pd.DataFrame:
           # Your analysis logic
           pass
   ```

2. **Add tests in `tests/`**
   ```python
   # tests/test_new_analytics.py
   import pytest
   from src.analytics.new_analytics import NewAnalytics
   
   def test_new_analytics():
       na = NewAnalytics()
       # Test implementation
   ```

3. **Update documentation**
   - Add to README.md
   - Update data_model.md if needed
   - Add API documentation if exposing via API

### Adding New SQL Views

1. **Create SQL file in appropriate directory**
   - `sql/kpis/` for KPI calculations
   - `sql/analytics/` for analytics views
   - `sql/transformations/` for transformation views

2. **Follow naming convention**
   - Prefix with view type: `kpi_`, `vw_analytics_`, `vw_`
   - Use snake_case
   - Add descriptive comments

3. **Test SQL views**
   ```sql
   -- Test your view
   SELECT * FROM your_new_view LIMIT 10;
   ```

### Adding ML Models

1. **Create model in `src/ml/`**
   ```python
   class NewModel:
       def __init__(self):
           self.model = None
       
       def train(self, data):
           # Training logic
           pass
       
       def predict(self, data):
           # Prediction logic
           pass
   ```

2. **Add model evaluation**
   - Include metrics (accuracy, precision, recall, etc.)
   - Save model to `models/` directory
   - Add model metadata

## Testing Guidelines

### Writing Tests

- **Unit tests**: Test individual functions/methods
- **Integration tests**: Test module interactions
- **End-to-end tests**: Test complete workflows

### Test Coverage

- Aim for >80% code coverage
- Focus on critical paths
- Test edge cases and error conditions

### Example Test

```python
import pytest
import pandas as pd
import numpy as np
from src.analytics.customer_analytics import CustomerIntelligence

def test_customer_360():
    # Arrange
    ci = CustomerIntelligence(data_dir="tests/fixtures")
    
    # Act
    result = ci.build_customer_360()
    
    # Assert
    assert isinstance(result, pd.DataFrame)
    assert 'customer_id' in result.columns
    assert len(result) > 0
```

## Documentation Guidelines

### Code Documentation

- Use docstrings for all modules, classes, and functions
- Follow Google docstring format
- Include parameter descriptions and return types

```python
def calculate_clv(customer_data: pd.DataFrame) -> pd.DataFrame:
    """Calculate Customer Lifetime Value.
    
    Args:
        customer_data: DataFrame containing customer transaction data
        
    Returns:
        DataFrame with CLV calculations and value tiers
        
    Raises:
        ValueError: If required columns are missing
    """
    pass
```

### README Updates

- Update README.md for significant features
- Include usage examples
- Update installation instructions if needed

### API Documentation

- Update `docs/api.md` for new endpoints
- Include request/response examples
- Document error codes

## Issue Reporting

### Bug Reports

Include:
- Description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, etc.)
- Screenshots if applicable

### Feature Requests

Include:
- Clear description of the feature
- Use case and benefits
- Possible implementation approach
- Alternative solutions considered

## Release Process

### Versioning

Follow semantic versioning (MAJOR.MINOR.PATCH):
- MAJOR: Breaking changes
- MINOR: New features (backwards compatible)
- PATCH: Bug fixes

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version number updated
- [ ] Git tag created
- [ ] Release notes published

## Code Review Guidelines

### For Reviewers

- Be constructive and specific
- Focus on code quality and maintainability
- Suggest improvements, don't just point out issues
- Approve when confident in the changes

### For Authors

- Respond to feedback promptly
- Explain reasoning for design decisions
- Be open to suggestions
- Update code based on feedback

## Performance Guidelines

### Database Queries

- Use indexes appropriately
- Avoid SELECT * in production
- Use EXPLAIN ANALYZE for slow queries
- Consider materialized views for heavy aggregations

### Python Code

- Use vectorized operations (Pandas/NumPy)
- Avoid loops where possible
- Use caching for expensive operations
- Profile performance bottlenecks

## Security Guidelines

- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user inputs
- Follow OWASP security guidelines
- Keep dependencies updated

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: For security issues

## Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the E-Commerce Intelligence Platform!
