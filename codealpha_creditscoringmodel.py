#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
columns = ['Status', 'Duration', 'Credit_history', 'Purpose', 'Credit_amount', 'Savings', 'Employment', 
           'Installment_rate', 'Personal_status', 'Other_parties', 'Residence_since', 'Property_magnitude', 
           'Age', 'Other_payment_plans', 'Housing', 'Existing_credits', 'Job', 'Num_dependents', 'Own_telephone', 
           'Foreign_worker', 'Class']
data = pd.read_csv(url, delimiter=' ', header=None, names=columns)


# In[3]:


data['Class'] = data['Class'].apply(lambda x: 0 if x == 2 else 1)  # Convert class to binary (1: good, 0: bad)
for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])


# In[4]:


X = data.drop('Class', axis=1)
y = data['Class']


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[6]:


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[9]:


print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')


# In[10]:


conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)


# In[11]:


importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




