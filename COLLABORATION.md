# Collaboration Guidelines

This document outlines best practices for collaborative development on the fake news detection project.

## Team Structure and Roles

### Person A: Data Specialist
**Primary Responsibilities:**
- Data collection and dataset preparation
- Exploratory data analysis (EDA)
- Data preprocessing and feature engineering
- Data quality assurance
- Creating data visualizations

**Key Files to Work On:**
- `notebooks/01_data_exploration.ipynb`
- `src/data/preprocessing.py`
- `src/features/`
- `data/` folder management

### Person B: Model Developer
**Primary Responsibilities:**
- Machine learning model implementation
- Model training and evaluation
- Hyperparameter tuning
- Web application development
- Model deployment

**Key Files to Work On:**
- `src/models/traditional_ml.py`
- `notebooks/03_traditional_ml.ipynb`
- `notebooks/04_deep_learning.ipynb`
- `web_app/app.py`

## Git Workflow

### Branch Strategy
```bash
# Main branches
main                    # Production-ready code
develop                # Integration branch

# Feature branches
feature/data-collection        # Person A's data work
feature/model-implementation   # Person B's model work
feature/web-app               # Web interface
feature/documentation         # Documentation updates
```

### Daily Workflow
1. **Start of Day:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout your-feature-branch
   git merge develop
   ```

2. **During Development:**
   ```bash
   # Make changes
   git add .
   git commit -m "feat: descriptive commit message"
   git push origin your-feature-branch
   ```

3. **End of Day:**
   ```bash
   # Create pull request on GitHub
   # Request review from teammate
   ```

### Commit Message Convention
```
type(scope): description

Types:
- feat: new feature
- fix: bug fix
- docs: documentation
- style: formatting
- refactor: code restructuring
- test: adding tests
- chore: maintenance

Examples:
feat(data): add text preprocessing pipeline
fix(model): resolve accuracy calculation bug
docs(readme): update installation instructions
```

## Code Standards

### Python Style Guide
- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Maximum line length: 88 characters (Black formatter)

### Example Function Documentation:
```python
def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Preprocess text for machine learning models.
    
    Args:
        text: Input text string to preprocess
        remove_stopwords: Whether to remove common English stopwords
        
    Returns:
        Processed text string with cleaning applied
        
    Example:
        >>> preprocess_text("This is a SAMPLE text!!!")
        "sample text"
    """
    # Implementation here
```

### File Organization
```python
# Import order:
# 1. Standard library imports
import os
import sys

# 2. Third-party imports
import pandas as pd
import numpy as np

# 3. Local imports
from src.data.preprocessing import TextPreprocessor
```

## Communication Protocols

### Daily Standups (15 minutes)
**Schedule:** Every day at 10:00 AM
**Format:** Quick check-in covering:
- What did you accomplish yesterday?
- What will you work on today?
- Any blockers or help needed?

### Weekly Reviews (45 minutes)
**Schedule:** Every Friday at 4:00 PM
**Agenda:**
- Code review of completed features
- Progress against project milestones
- Plan for next week
- Technical discussions and decisions

### Communication Channels
- **Slack/Discord:** Daily quick questions and updates
- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Technical discussions and Q&A
- **Video Calls:** Complex problem-solving sessions

## Code Review Process

### Before Creating Pull Request:
1. Test your code locally
2. Run linters and formatters
3. Update documentation if needed
4. Write descriptive PR description

### Pull Request Template:
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Code tested locally
- [ ] All existing tests pass
- [ ] New tests added if applicable

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No merge conflicts
```

### Review Guidelines:
- Review code within 24 hours
- Be constructive and specific in feedback
- Approve only after thorough testing
- Discuss complex issues in person/video call

## File Sharing and Backup

### Large Files (>100MB):
- Use Git LFS for model files
- Store datasets in cloud storage (Google Drive, Dropbox)
- Never commit large datasets to Git

### Backup Strategy:
- Daily: Push code to GitHub
- Weekly: Backup datasets and models to cloud
- Before deadlines: Complete project backup

## Testing Strategy

### Test Types:
1. **Unit Tests:** Test individual functions
2. **Integration Tests:** Test component interactions
3. **End-to-End Tests:** Test complete workflows

### Testing Schedule:
- Write tests alongside new features
- Run full test suite before merging to develop
- Add tests for bug fixes

### Example Test Structure:
```python
def test_text_preprocessing():
    """Test text preprocessing pipeline."""
    preprocessor = TextPreprocessor()
    
    # Test case 1: Normal text
    result = preprocessor.preprocess_text("Hello World!")
    assert result == "hello world"
    
    # Test case 2: Empty text
    result = preprocessor.preprocess_text("")
    assert result == ""
```

## Troubleshooting Common Issues

### Merge Conflicts:
```bash
# 1. Pull latest changes
git pull origin develop

# 2. Resolve conflicts manually
# Edit conflicted files

# 3. Mark as resolved
git add .
git commit -m "resolve: merge conflicts"
```

### Environment Issues:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install -r requirements.txt
```

### GitHub Issues:
- Create issue for bugs or questions
- Use labels: bug, enhancement, question, help-wanted
- Assign issues to responsible person
- Link PRs to related issues

## Project Milestones

### Week 1-2: Foundation
- [ ] Data collection complete
- [ ] EDA notebook finished
- [ ] Preprocessing pipeline ready
- [ ] Basic project structure established

### Week 3-4: Traditional ML
- [ ] Baseline models implemented
- [ ] Model evaluation framework ready
- [ ] Performance comparison complete
- [ ] Initial web interface prototype

### Week 5-6: Deep Learning
- [ ] Neural network models implemented
- [ ] Advanced techniques explored
- [ ] Model comparison updated
- [ ] Web interface enhanced

### Week 7-8: Finalization
- [ ] Best model selected and tuned
- [ ] Web application polished
- [ ] Documentation complete
- [ ] Presentation prepared

## Emergency Contacts

- **Technical Issues:** Create GitHub issue and notify teammate
- **Deadline Pressure:** Communicate early and reallocate tasks
- **Personal Issues:** Notify teammate ASAP for task coverage

Remember: Communication is key to successful collaboration!