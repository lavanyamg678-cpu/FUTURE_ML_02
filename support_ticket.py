import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import nltk
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
# Load dataset
df = pd.read_csv("data/customer_support_tickets.csv")

# Display first 5 rows
print(df.head())

# Display dataset information
print(df.info())

# Display missing values
print(df.isnull().sum())
print(df.head())
print(df.info())
print(df.isnull().sum())
# Define stop words
stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    words = text.split()  # Split into words
    words = [word for word in words if word not in stop_words]  # Remove stop words
    return " ".join(words)

# Clean the Ticket Description column
df['Cleaned_Text'] = df['Ticket Description'].apply(clean_text)

# Display cleaned text
print(df[['Ticket Description', 'Cleaned_Text']].head())
# Convert cleaned text into TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df['Cleaned_Text'])

print("Shape of TF-IDF Matrix:", X.shape)
print(df.columns)
# Features and Target
X = vectorizer.fit_transform(df['Cleaned_Text'])
y = df['Ticket Type']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)
model = MultinomialNB()
model.fit(X_train, y_train)

print("Model trained successfully!")
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
# Confusion Matrix for Ticket Type Classification
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Ticket Type Classification")
plt.show()
print("\n----- Priority Prediction -----")

# Target for priority prediction
y_priority = df['Ticket Priority']

# Split the data
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

# Train the model
priority_model = MultinomialNB()
priority_model.fit(X_train_p, y_train_p)

# Make predictions
priority_pred = priority_model.predict(X_test_p)

# Evaluate
print("Priority Prediction Accuracy:",
      accuracy_score(y_test_p, priority_pred))

print("\nPriority Classification Report:")
print(classification_report(y_test_p, priority_pred))
# Confusion Matrix for Priority Prediction
cm_priority = confusion_matrix(y_test_p, priority_pred)

disp_priority = ConfusionMatrixDisplay(
    confusion_matrix=cm_priority,
    display_labels=priority_model.classes_
)

disp_priority.plot(cmap='Greens')
plt.title("Confusion Matrix - Ticket Priority Prediction")
plt.show()