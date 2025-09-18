# Fake News Detection Project

This project uses a Transformer-based deep learning model to classify news articles as "Real" or "Fake". It includes scripts for data preprocessing, model training, and a simple web interface built with Streamlit.

## Project Structure

```
/
├── .gitignore
├── data/
├── notebooks/
├── saved_models/
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── data_preprocessing.py
│   ├── predict.py
│   └── train.py
└── README.md
```

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aashishlalwani/Fakenewsnew.git
    cd Fakenewsnew
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run src/app.py
    ```
