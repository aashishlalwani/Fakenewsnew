"""
Traditional machine learning models for fake news detection.

This module contains implementations of various traditional ML algorithms
including Naive Bayes, Logistic Regression, SVM, and Random Forest.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib


class TraditionalMLModels:
    """
    A class to handle traditional machine learning models for fake news detection.
    """
    
    def __init__(self):
        """Initialize the models and vectorizers."""
        self.models = {
            'naive_bayes': MultinomialNB(),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'svm': SVC(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        self.vectorizers = {
            'tfidf': TfidfVectorizer(max_features=10000, stop_words='english'),
            'count': CountVectorizer(max_features=10000, stop_words='english')
        }
        
        self.pipelines = {}
        self.trained_models = {}
    
    def create_pipelines(self) -> None:
        """Create ML pipelines combining vectorizers and models."""
        for vectorizer_name, vectorizer in self.vectorizers.items():
            for model_name, model in self.models.items():
                pipeline_name = f"{vectorizer_name}_{model_name}"
                self.pipelines[pipeline_name] = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', model)
                ])
    
    def train_model(self, 
                    X_train: pd.Series, 
                    y_train: pd.Series,
                    pipeline_name: str) -> None:
        """
        Train a specific pipeline.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            pipeline_name: Name of the pipeline to train
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        print(f"Training {pipeline_name}...")
        self.pipelines[pipeline_name].fit(X_train, y_train)
        self.trained_models[pipeline_name] = self.pipelines[pipeline_name]
        print(f"{pipeline_name} trained successfully!")
    
    def train_all_models(self, 
                         X_train: pd.Series, 
                         y_train: pd.Series) -> None:
        """
        Train all pipelines.
        
        Args:
            X_train: Training text data
            y_train: Training labels
        """
        self.create_pipelines()
        
        for pipeline_name in self.pipelines.keys():
            self.train_model(X_train, y_train, pipeline_name)
    
    def evaluate_model(self, 
                       X_test: pd.Series, 
                       y_test: pd.Series,
                       pipeline_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            pipeline_name: Name of the pipeline to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if pipeline_name not in self.trained_models:
            raise ValueError(f"Model {pipeline_name} is not trained yet")
        
        model = self.trained_models[pipeline_name]
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred
        }
    
    def evaluate_all_models(self, 
                            X_test: pd.Series, 
                            y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test text data
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation results for all models
        """
        results = {}
        
        for pipeline_name in self.trained_models.keys():
            print(f"Evaluating {pipeline_name}...")
            results[pipeline_name] = self.evaluate_model(X_test, y_test, pipeline_name)
            print(f"Accuracy: {results[pipeline_name]['accuracy']:.4f}")
            print("-" * 50)
        
        return results
    
    def cross_validate_model(self, 
                             X_train: pd.Series, 
                             y_train: pd.Series,
                             pipeline_name: str,
                             cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            pipeline_name: Name of the pipeline to cross-validate
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline {pipeline_name} not found")
        
        pipeline = self.pipelines[pipeline_name]
        
        # Perform cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'all_scores': scores
        }
    
    def hyperparameter_tuning(self, 
                              X_train: pd.Series, 
                              y_train: pd.Series,
                              model_type: str = 'logistic_regression') -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            model_type: Type of model to tune
            
        Returns:
            Dictionary with best parameters and scores
        """
        # Define parameter grids
        param_grids = {
            'logistic_regression': {
                'vectorizer__max_features': [5000, 10000, 20000],
                'vectorizer__ngram_range': [(1, 1), (1, 2)],
                'classifier__C': [0.1, 1, 10]
            },
            'svm': {
                'vectorizer__max_features': [5000, 10000],
                'vectorizer__ngram_range': [(1, 1), (1, 2)],
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf']
            },
            'random_forest': {
                'vectorizer__max_features': [5000, 10000],
                'vectorizer__ngram_range': [(1, 1), (1, 2)],
                'classifier__n_estimators': [50, 100, 200],
                'classifier__max_depth': [10, 20, None]
            }
        }
        
        if model_type not in param_grids:
            raise ValueError(f"Hyperparameter tuning not available for {model_type}")
        
        # Create pipeline
        pipeline_name = f"tfidf_{model_type}"
        if pipeline_name not in self.pipelines:
            self.create_pipelines()
        
        pipeline = self.pipelines[pipeline_name]
        param_grid = param_grids[model_type]
        
        # Perform grid search
        print(f"Performing hyperparameter tuning for {model_type}...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=3, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def save_model(self, pipeline_name: str, filepath: str) -> None:
        """
        Save a trained model to disk.
        
        Args:
            pipeline_name: Name of the pipeline to save
            filepath: Path to save the model
        """
        if pipeline_name not in self.trained_models:
            raise ValueError(f"Model {pipeline_name} is not trained yet")
        
        joblib.dump(self.trained_models[pipeline_name], filepath)
        print(f"Model {pipeline_name} saved to {filepath}")
    
    def load_model(self, filepath: str, pipeline_name: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            pipeline_name: Name to assign to the loaded pipeline
        """
        model = joblib.load(filepath)
        self.trained_models[pipeline_name] = model
        print(f"Model loaded from {filepath} as {pipeline_name}")
    
    def get_feature_importance(self, pipeline_name: str, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for tree-based models.
        
        Args:
            pipeline_name: Name of the pipeline
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature names and importance scores
        """
        if pipeline_name not in self.trained_models:
            raise ValueError(f"Model {pipeline_name} is not trained yet")
        
        model = self.trained_models[pipeline_name]
        
        # Check if model has feature importance
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            feature_names = model.named_steps['vectorizer'].get_feature_names_out()
            importance_scores = model.named_steps['classifier'].feature_importances_
            
            # Create DataFrame
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_scores
            }).sort_values('importance', ascending=False).head(top_n)
            
            return feature_importance
        else:
            print(f"Model {pipeline_name} does not have feature importance")
            return None


def compare_model_performance(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare performance of multiple models.
    
    Args:
        results: Dictionary containing evaluation results for multiple models
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision (FAKE)': result['classification_report']['FAKE']['precision'],
            'Recall (FAKE)': result['classification_report']['FAKE']['recall'],
            'F1-Score (FAKE)': result['classification_report']['FAKE']['f1-score'],
            'Precision (REAL)': result['classification_report']['REAL']['precision'],
            'Recall (REAL)': result['classification_report']['REAL']['recall'],
            'F1-Score (REAL)': result['classification_report']['REAL']['f1-score']
        })
    
    return pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)


if __name__ == "__main__":
    # Example usage
    print("Traditional ML Models for Fake News Detection")
    print("This module provides various ML algorithms for text classification")