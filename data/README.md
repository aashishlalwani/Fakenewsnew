# Data Collection Guide

This document provides guidance on collecting and preparing datasets for the fake news detection project.

## Recommended Datasets

### 1. LIAR Dataset
- **Source**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
- **Description**: Contains 12.8K manually labeled short statements from POLITIFACT.COM
- **Labels**: 6 categories (pants-fire, false, barely-true, half-true, mostly-true, true)
- **Format**: TSV files
- **Best for**: Statement-level fact-checking

### 2. Fake News Detection Dataset (Kaggle)
- **Source**: https://www.kaggle.com/c/fake-news/data
- **Description**: News articles with binary labels (FAKE/REAL)
- **Size**: ~20K articles
- **Format**: CSV
- **Best for**: Article-level classification

### 3. ISOT Fake News Dataset
- **Source**: https://www.uvic.ca/engineering/ece/isot/datasets/fake-news-detection/
- **Description**: Collection of real and fake news articles
- **Size**: 44K articles (21K real, 23K fake)
- **Format**: CSV
- **Best for**: Balanced binary classification

### 4. FakeNewsNet
- **Source**: https://github.com/KaiDMML/FakeNewsNet
- **Description**: Social media fake news dataset with social context
- **Features**: News content, social engagement, user profiles
- **Best for**: Social media analysis

## Data Collection Steps

### Step 1: Download Primary Dataset
1. Choose one of the recommended datasets above
2. Download and extract to `data/raw/` folder
3. Read the documentation and understand the format

### Step 2: Data Validation
```python
import pandas as pd

# Load the dataset
df = pd.read_csv('data/raw/your_dataset.csv')

# Check basic information
print(df.info())
print(df.head())
print(df['label'].value_counts())
```

### Step 3: Data Cleaning Checklist
- [ ] Remove duplicates
- [ ] Handle missing values
- [ ] Standardize label formats
- [ ] Remove very short articles (< 50 words)
- [ ] Check for encoding issues

### Step 4: Create Data Splits
```python
# Split data into train/val/test
from sklearn.model_selection import train_test_split

# 70% train, 15% val, 15% test
train, temp = train_test_split(df, test_size=0.3, random_state=42, stratify=df['label'])
val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['label'])

# Save splits
train.to_csv('data/processed/train.csv', index=False)
val.to_csv('data/processed/val.csv', index=False)
test.to_csv('data/processed/test.csv', index=False)
```

## Alternative Data Sources

### News APIs (for real-time data)
- **NewsAPI**: https://newsapi.org/
- **Guardian API**: https://open-platform.theguardian.com/
- **New York Times API**: https://developer.nytimes.com/

### Social Media APIs
- **Twitter API**: For tweet analysis
- **Reddit API**: For discussion analysis

## Data Storage Structure

```
data/
├── raw/                    # Original downloaded datasets
│   ├── liar_dataset.csv
│   ├── fake_news_kaggle.csv
│   └── isot_dataset.csv
├── processed/              # Cleaned and split datasets
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── processed_full.csv
└── external/               # Additional data sources
    ├── news_api_data.json
    └── social_media_data.json
```

## Data Privacy and Ethics

### Important Considerations:
1. **Respect Copyright**: Ensure you have rights to use the data
2. **Privacy**: Remove personal information if present
3. **Bias**: Be aware of potential biases in the dataset
4. **Fairness**: Ensure balanced representation across different sources

### Data Usage Guidelines:
- Use datasets only for educational/research purposes
- Don't redistribute copyrighted content
- Acknowledge data sources in your project
- Follow the original dataset's license terms

## Quality Checks

Before proceeding with modeling, perform these checks:

```python
def data_quality_check(df):
    """Perform basic data quality checks."""
    print("=== Data Quality Report ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {df.columns.tolist()}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Text length statistics
    if 'text' in df.columns:
        text_lengths = df['text'].str.len()
        print(f"\nText length stats:")
        print(f"Mean: {text_lengths.mean():.2f}")
        print(f"Median: {text_lengths.median():.2f}")
        print(f"Min: {text_lengths.min()}")
        print(f"Max: {text_lengths.max()}")

# Run quality check
data_quality_check(df)
```

## Troubleshooting Common Issues

### 1. Encoding Problems
```python
# Try different encodings
df = pd.read_csv('data.csv', encoding='utf-8')
# or
df = pd.read_csv('data.csv', encoding='latin-1')
```

### 2. Large Files
```python
# Read in chunks for large files
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

### 3. Memory Issues
```python
# Optimize data types
df['label'] = df['label'].astype('category')
df['text'] = df['text'].astype('string')
```

## Next Steps

After collecting your data:
1. Run the data exploration notebook (`01_data_exploration.ipynb`)
2. Implement preprocessing using the provided tools
3. Start with traditional ML models
4. Progress to deep learning approaches