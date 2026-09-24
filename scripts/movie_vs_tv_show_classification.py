#!/usr/bin/env python3
"""
Netflix Movie vs TV Show Classification
A simple script version of the classification notebook.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Netflix Movie vs TV Show Classification")
    print("=" * 50)

    # Load dataset
    df = pd.read_csv('Data/netflix_titles.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Map target variable
    df['type_encoded'] = df['type'].map({'Movie': 0, 'TV Show': 1})
    print(f"Target distribution: Movie={sum(df['type_encoded']==0)}, TV Show={sum(df['type_encoded']==1)}")

    # Feature selection
    selected_features = [
        'release_year', 'rating', 'duration', 'listed_in',
        'country', 'date_added', 'director'
    ]

    # Data preprocessing (simplified version)
    df_processed = df.copy()

    # Handle missing values
    for feature in selected_features:
        if df_processed[feature].isnull().sum() > 0:
            if df_processed[feature].dtype == 'object':
                df_processed[feature] = df_processed[feature].fillna('Unknown')
            else:
                df_processed[feature] = df_processed[feature].fillna(df_processed[feature].median())

    # Feature engineering
    # Duration - extract numeric
    df_processed['duration_numeric'] = df_processed['duration'].str.extract(r'(\d+)').astype(float)
    df_processed['duration_numeric'] = df_processed['duration_numeric'].fillna(df_processed['duration_numeric'].median())

    # Date added - extract year
    df_processed['date_added'] = pd.to_datetime(df_processed['date_added'], errors='coerce')
    df_processed['date_added_year'] = df_processed['date_added'].dt.year
    df_processed['date_added_year'] = df_processed['date_added_year'].fillna(df_processed['date_added_year'].median())

    # Director - simple features
    df_processed['has_director'] = (df_processed['director'] != 'Unknown').astype(int)
    df_processed['director_count'] = df_processed['director'].apply(
        lambda x: len(str(x).split(', ')) if x != 'Unknown' else 0
    )

    # Encode categorical features
    label_encoders = {}
    for feature in ['rating', 'listed_in', 'country']:
        le = LabelEncoder()
        df_processed[feature] = le.fit_transform(df_processed[feature])
        label_encoders[feature] = le

    # Prepare final feature set
    feature_cols = [
        'release_year', 'duration_numeric', 'date_added_year',
        'rating', 'listed_in', 'country',
        'has_director', 'director_count'
    ]

    X = df_processed[feature_cols]
    y = df_processed['type_encoded']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = ['release_year', 'duration_numeric', 'date_added_year']
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    # Find best model
    best_model = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest Model: {best_model}")
    print(f"Accuracy: {results[best_model]['accuracy']:.4f}")

    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv('results/model_comparison.csv')
    print("\nResults saved to results/model_comparison.csv")

    print("\nClassification completed successfully!")

if __name__ == "__main__":
    main()