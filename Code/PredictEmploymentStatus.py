
# coding: utf-8

# # Data Preparation 

# In[1]:


## Import data from file
# Pandas is used for data manipulation
import pandas as pd

# Read in data as pandas dataframe and display first 5 rows
training_data = pd.read_csv('train.csv')
testing_data = pd.read_csv('test.csv')


# In[2]:


training_data.head(5)


# In[3]:


testing_data.head(5)


# In[4]:


training_data.info()


# In[5]:


testing_data.info()


# In[6]:


training_data = training_data.replace('9th grade',9)
training_data = training_data.replace('10th grade',10)
training_data = training_data.replace('11th grade',11)
training_data = training_data.replace('12th grade',12)
training_data = training_data.replace('High School',13)
training_data = training_data.replace('Associate Degree',14)
training_data = training_data.replace('Bachelor',15)
training_data = training_data.replace('Master',16)
training_data = training_data.replace('Doctoral Degree',17)
training_data = training_data.replace('Prof. Degree',18)
training_data = training_data.replace('Some College',19)

training_data = training_data.replace('Unemployed',3)
training_data = training_data.replace('Employed',4)
training_data = training_data.replace('Not in labor force',5)

training_data = training_data.replace('Male',1)
training_data = training_data.replace('Female',2)

training_data = training_data.drop('Age Range', axis = 1)
training_data = training_data.drop('Id', axis = 1)
training_data = training_data.drop('Total', axis = 1)

training_data.info()


# In[7]:


testing_data.info()


# In[8]:



testing_data = testing_data.replace('9th grade',9)
testing_data = testing_data.replace('10th grade',10)
testing_data = testing_data.replace('11th grade',11)
testing_data = testing_data.replace('12th grade',12)
testing_data = testing_data.replace('High School',13)
testing_data = testing_data.replace('Associate Degree',14)
testing_data = testing_data.replace('Bachelor',15)
testing_data = testing_data.replace('Master',16)
testing_data = testing_data.replace('Doctoral Degree',17)
testing_data = testing_data.replace('Prof. Degree',18)
testing_data = testing_data.replace('Some College',19)

testing_data = testing_data.replace('Male',1)
testing_data = testing_data.replace('Female',2)

testing_data = testing_data.drop('Age Range', axis = 1)
testing_data = testing_data.drop('id', axis = 1)
testing_data = testing_data.drop('Total', axis = 1)

testing_data.info()


# In[9]:


# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(training_data['Employment Status'])

# Remove the labels from the features
# axis 1 refers to the columns
features = training_data.drop('Employment Status', axis = 1)

# Saving feature names for later use
feature_list = list(features.columns)

# Convert to numpy array
features = np.array(features)

#######################################################

# Labels are the values we want to predict
labels_test = np.array(testing_data['Employment Status'])

features_test = testing_data.drop('Employment Status', axis = 1)

# Saving feature names for later use
feature_list_test = list(features_test.columns)

# Convert to numpy array
features_test = np.array(features_test)


# In[10]:


feature_list_test


# ## Training and Testing Sets

# In[11]:


train_features = features
test_features = features_test
train_labels = labels
test_labels = labels_test


# In[12]:


# Scaling the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(train_features)
train_features = scaler.transform(train_features)
test_features = scaler.transform(test_features)


# In[13]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[14]:


# Create the model and tune the the parameters
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(21, 21, 21),
                   activation='tanh', max_iter=2000)
mlp.fit(train_features, train_labels)
predictions = mlp.predict(test_features)


# In[15]:


predictions = np.array(predictions)


# In[16]:


employment_status = pd.DataFrame(predictions)


# In[17]:


employment_status = employment_status.replace(3, 'Unemployed')
employment_status = employment_status.replace(4, 'Employed')
employment_status = employment_status.replace(5, 'Not in labor force')
employment_status.rename(columns={0: 'Employment Status'}, inplace=True)


# In[18]:


testing_data = testing_data.drop('Employment Status', axis = 1)


# In[19]:


testing_data = testing_data.join(employment_status)


# In[20]:


testing_data


# In[21]:


feature_list_test.append('Employment Status')
testing_data.to_csv('predicted_Employment_Status.csv', columns = feature_list_test)


# ## Variable Importances

# In[22]:


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(train_features, train_labels);

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

