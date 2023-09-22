#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as  np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


data_set=pd.read_csv("heart_statlog_cleveland_hungary_final.csv")
print("(Rows, columns): " + str(data_set.shape))
data_set.columns


# In[3]:


data_set.head(5)


# In[4]:


print(data_set.isna().sum())


# In[5]:


dup_row=data_set[data_set.duplicated()]
print("Duplicate Rows: \n{}".format(dup_row))


# In[6]:


DF_RM_DUP = data_set.drop_duplicates(keep='first')
print('\n\nDuplicate value removed from DataFrame' )
data_set=DF_RM_DUP


# In[7]:


data_set.describe()


# In[8]:


data_set.hist(figsize=(15,15))


# In[9]:


data_set['target'].value_counts()


# In[ ]:





# In[10]:


corr = data_set.corr()
plt.subplots(figsize=(15,10))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap=sns.diverging_palette(220, 20, as_cmap=True))


# In[11]:


x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values      #dependent


# In[12]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)


# In[13]:


x_train.shape


# In[14]:


x_test.shape


# In[15]:


y_train.shape


# In[16]:


y_test.shape


# In[17]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[18]:


# Initialize an empty list to store accuracy values for different settings
accuracy_values = []

# Replace the loop range and parameters with the ones you want to experiment with
for n_trees in range(1,100, 5):
    # Initialize the Random Forest classifier with the desired settings
    rf_clf = RandomForestClassifier(n_estimators=n_trees, random_state=1)

    # Train the classifier on the training data
    rf_clf.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = rf_clf.predict(x_test)

    # Calculate accuracy and append to the list
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_values.append(accuracy)


# In[19]:


import numpy as np
import matplotlib.pyplot as plt

# Sample data for illustration purposes
x_values = range(1, 100, 5) 
# Plot the accuracy values
plt.plot(x_values, accuracy_values, marker='o', label='Accuracy')

# The y-coordinate at which you want to draw the parallel line
y_value = np.max(accuracy_values)

# Find the indices where accuracy_values matches y_value
indices = np.where(accuracy_values == y_value)

# Get the corresponding x-values from x_values using the indices
x_values_at_y_value = np.array(x_values)[indices]

# Plot the parallel line
plt.axhline(y=y_value, color='red', linestyle='--', label='Parallel Line')
for x_value in x_values_at_y_value:
    plt.axvline(x=x_value, color='red', linestyle='--', label='Parallel Line')

# Set labels and title
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Number of Trees for Random Forest')
# Display the plot
plt.grid(True)
plt.show()


print("x-values at y-value:", x_values_at_y_value)


# In[20]:


model = RandomForestClassifier(n_estimators=x_value,random_state=1)# get instance of model
model.fit(x_train, y_train) # Train/Fit model 


# In[21]:


y_pred = model.predict(x_test) # get y predictions
val=(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
print(val[:5])
#First value represents our predicted value, Second value represents our actual value.


# In[22]:


cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# In[23]:


# get importance
importance = model.feature_importances_
print('Feature\t Score')
# summarize feature importance
for i,v in enumerate(importance):
    print(i,'\t',v)


# In[24]:


import matplotlib.pyplot as plt
index= data_set.columns[:-1]
importance = pd.Series(model.feature_importances_)
y_pos=np.arange(len(index))
plt.barh(y_pos,importance)
plt.yticks(y_pos,index)
#importance.nlargest(13).plot(kind='bar', colormap='winter')


# In[25]:


#Predictibg from the model
li=[20,1,1,180,192,1,0,90,0,1,1]
li2=[57,0,3,190,236,1,0,174,0,1,2]
p=model.predict(sc.transform([li2]))
if(p==1):
    print("Heart Patient")
else:
    print("Healthy")

