#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing libraries 
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


#INITIAL IMPORT AND DATA EXPLORATION


# In[3]:


#import data set and data set exploration 
file_path = '/Users/vincent/Desktop/Python/Deep Learning/TelcoCustomerChurnRate.csv'
churn_data = pd.read_csv(file_path, index_col=0)


# In[4]:


churn_data


# In[5]:


churn_data.shape


# In[6]:


churn_data.info()


# In[7]:


churn_data.columns.values


# In[8]:


#change dtype of TotalCharges to numeric
churn_data['TotalCharges'] = pd.to_numeric(churn_data.TotalCharges, errors='coerce')

#check for missing values 
churn_data.isnull().sum()


# In[9]:


churn_data.dtypes


# In[10]:


#drop the missing values as the data set is large enough 
churn_data.dropna(inplace=True)


# In[11]:


#check again is missing values remain 
churn_data.isnull().sum()


# In[12]:


#CREATE ENCODER


# In[13]:


for column in churn_data.columns:
    unique_values = churn_data[column].unique()
    print(f"Unique values in {column}: {unique_values}")


# In[14]:


# List of categorical columns for encoding
categorical_columns = [
    'gender', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn'
]

# Create encoding dictionaries for each categorical column
encoding_dictionaries = {}
for column in categorical_columns:
    unique_values = list(churn_data[column].unique())
    encoding_dictionaries[column] = {val: i for i, val in enumerate(unique_values)}

# Apply the encoding dictionaries to the DataFrame
for column in categorical_columns:
    churn_data[column] = churn_data[column].map(encoding_dictionaries[column])

# Verify the encoding
churn_data.head()


# In[15]:


encoding_dictionaries


# In[16]:


#check all columns have numerical data
churn_data.info()


# In[17]:


#CREATE THE NEURAL NETWORK MODEL TO PREDICT CHURN RATES


# In[18]:


#Splitting the data into features and target variable
X = churn_data.drop('Churn', axis=1)
y = churn_data['Churn']

#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Normalizing the feature data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Defining the model architecture 
model = Sequential([
    Input(shape=(X_train.shape[1],)), 
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

#Compiling the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the model, hyperparameters where determined by testing different values
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)])

#Testing on the test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

