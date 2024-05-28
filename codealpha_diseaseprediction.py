#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
column_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
data = pd.read_csv(url, names=column_names)


print(data.head()) 


# In[12]:


data.replace('?', pd.NA, inplace=True)


data = data.apply(pd.to_numeric, errors='coerce')


data.dropna(inplace=True)


print(data.info())


# In[13]:


X = data.drop("target", axis=1)
y = data["target"]


print(y.value_counts())


# In[14]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)


model.fit(X_train, y_train)


# In[15]:


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

