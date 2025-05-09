import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Path to your dataset
DATA_PATH = r"C:/Users/user/Downloads/archive"

# Parameters
sr = 22050  # Sample rate
n_mfcc = 13

# Extract features and labels
def extract_features_and_labels(data_path):
    features = []
    labels = []
    existing_labels = [label for label in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, label))]

    print(f"Checking files in {data_path}")
    for label in existing_labels:
        class_dir = os.path.join(data_path, label)
        print(f"Processing class: {label}")
        for filename in os.listdir(class_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(class_dir, filename)
                try:
                    y, _ = librosa.load(file_path, sr=sr)
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    mfcc_mean = np.mean(mfcc.T, axis=0)
                    features.append(mfcc_mean)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return np.array(features), np.array(labels)

# Main workflow
X, y = extract_features_and_labels(DATA_PATH)

if len(X) == 0:
    print("No features were extracted. Please check your dataset.")
else:
    print(f"Extracted {len(X)} samples with {X.shape[1]} features each.")

    # Encode only existing labels
    le = LabelEncoder()
    le.fit(y)  # Fit only on labels that were actually found
    y_encoded = le.transform(y)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

    # Train model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred = clf.predict(X_test)

    # Evaluate
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=np.unique(y_test), target_names=le.inverse_transform(np.unique(y_test))))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
