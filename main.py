import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import os

# Load dataset
data = pd.read_csv("data/synthetic_mbti_dataset.csv")

# Features and dimensions
X = data[[f"Q{i}" for i in range(1, 16)]]
dimensions = ["IE", "NS", "TF", "JP"]

models = {}

# Create a directory to save models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Train a model for each dimension
for dim in dimensions:
    print(f"\n📊 Training model for {dim} dimension...")
    y = data[dim]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use SVM with standard scaling
    model = make_pipeline(StandardScaler(), SVC(kernel='rbf', C=1.0, gamma='scale'))

    try:
        # Train the model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate accuracy
        acc = accuracy_score(y_test, y_pred) * 100
        print(f"\n📊 Accuracy for {dim}: {acc:.2f}%")
        print(classification_report(y_test, y_pred))

        # Save the trained model
        model_filename = f"models/{dim}_model.pkl"
        joblib.dump(model, model_filename)
        models[dim] = model

    except Exception as e:
        print(f"❌ Error training model for {dim}: {e}")

print("\n✅ All models saved successfully!")
