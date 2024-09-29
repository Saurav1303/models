
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pandas as pd

# Load the uploaded dataset
file_path = '/mnt/data/fiverrmain.xlsx'
df = pd.read_excel(file_path)

# Display the first few rows to understand its structure
df.head()


# 1. Data Preprocessing
# Handle missing values for 'Provider Rating' and 'Rating Count'
df['Provider Rating'].fillna(df['Provider Rating'].mean(), inplace=True)
df['Rating Count'] = pd.to_numeric(df['Rating Count'], errors='coerce')
df['Rating Count'].fillna(df['Rating Count'].mean(), inplace=True)

# Label encode categorical columns
le = LabelEncoder()
df['Provider KYC Verified'] = le.fit_transform(df['Provider  KYC Verified'])
df['Gig Value'] = df['Gig Value'].str.replace('US$', '').astype(float)

# TF-IDF for text features (Gig Title, Gig Description, Review 1)
tfidf = TfidfVectorizer(max_features=100)

gig_title_tfidf = tfidf.fit_transform(df['Gig Title']).toarray()
gig_desc_tfidf = tfidf.fit_transform(df['Gig Description']).toarray()
review_tfidf = tfidf.fit_transform(df['Review 1']).toarray()

# Combine all features into one matrix
X = pd.concat([pd.DataFrame(gig_title_tfidf), pd.DataFrame(gig_desc_tfidf), 
               pd.DataFrame(review_tfidf), df[['Provider KYC Verified', 'Gig Value']]], axis=1)
y = df['Provider Rating'] >= 4.5  # Classify if the provider rating is high or not

# 2. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Classification Model
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# 4. Model Evaluation
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
