#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os                                                  #for os dependent functionality
import tarfile                                             #to read and write tar archives     
import pandas as pd                                        #Pandas library imported under alias pd
import numpy as np                                         #Numpy library imported under alias np


# In[2]:


'''six - six is a Python 2 and 3 compatibility library. 
It provides utility functions for smoothing over the differences 
between the Python versions with the goal of writing Python code 
that is compatible on both Python versions.

six.moves - Python 3 reorganized the standard library and moved several functions to different modules. 
Six provides a consistent interface to them through the six.moves module.

urllib - This module provides a high-level interface for fetching data across the World Wide Web.'''

from six.moves import urllib 


# In[3]:


#........................................FETCHING DATA.............................................................................

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

'''creating a datasets/housing directory in your workspace, 
downloading the housing.tgz file, 
and extracting the housing.csv from it in this directory.'''

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):                                     #checks if the dir already exists.
        os.makedirs(housing_path)                                           #if does not exist, makes new dir.
    tgz_path = os.path.join(housing_path, "housing.tgz")                    #joins the path to get a new path
    urllib.request.urlretrieve(housing_url, tgz_path)                       #Copy a network object denoted by a URL to a local file.
    housing_tgz = tarfile.open(tgz_path)                                    #returns a tarfile object for the path.
    housing_tgz.extractall(path=housing_path)                               #extract the files
    housing_tgz.close()


# In[4]:


#....................................................LOADING DATA................................................................

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# In[5]:


fetch_housing_data()                                                    #fetch the data


# In[6]:


housing = load_housing_data()                                           #load the data in a dataframe 


# In[7]:


housing.dtypes


# In[8]:


housing.head()


# In[9]:


housing.info()                                                   #quick description of data


# In[10]:


housing['ocean_proximity'].value_counts()                        #how many districts belong to each category


# In[11]:


housing.describe()                                                     #summary of numerical attributes


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


housing.hist(bins=50, figsize=(20,15))
plt.show()


# In[14]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# In[15]:


train_set


# In[16]:


test_set


# In[17]:


housing=train_set.copy()


# In[18]:


housing


# In[19]:


housing.plot(kind="scatter", x="longitude", y="latitude")


# In[20]:


housing.plot(kind="scatter", x="longitude", y="latitude",alpha=0.1)


# In[21]:


corr_matrix = housing.corr()


# In[22]:


corr_matrix["median_house_value"].sort_values(ascending=False)  #how much each attribute correlates with the median house value


# In[23]:


housing.plot(kind="scatter", x="median_income", y="median_house_value",alpha=0.1) #median_income is highly correlated


# In[24]:


#create more meaningful attributes
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]


# In[25]:


housing


# In[26]:


corr_matrix = housing.corr()


# In[27]:


corr_matrix["median_house_value"].sort_values(ascending=False)


# In[28]:


#...............SEPARATE THE PREDICTORS AND THE LABELS.............................................

housing = train_set.drop("median_house_value", axis=1)   #does not affect the train_set.Only a copy is created.
housing_labels = train_set["median_house_value"].copy()


# In[29]:


housing


# In[30]:


train_set


# In[31]:


''' create an Imputer instance, specifying that you want to replace
    each attribute’s missing values with the median of that attribute'''

from sklearn.impute import SimpleImputer  
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)           #median cannot be computed on categorical attributes
imputer.fit(housing_num)                                        # fit the imputer instance to the training data


# In[32]:


housing_num


# In[33]:


imputer.statistics_             #calculates median of each attribute


# In[34]:


X = imputer.transform(housing_num)      #The result is a plain Numpy array containing the transformed features


# In[35]:


X


# In[36]:


housing_tr = pd.DataFrame(X, columns=housing_num.columns) #put it back into a Pandas DataFrame


# In[37]:


#..........................ENCODING CATEGORICAL ATTRIBUTE.........................................

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat.values.reshape(-1,1))


# In[38]:


housing_cat_1hot


# In[39]:


housing_cat_1hot.toarray()


# In[40]:


#................................................CUSTOM TRANSFORMERS........................................................

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
   
    def fit(self, X, y=None):
        return self # nothing else to do
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# In[41]:


X


# In[42]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[43]:


housing_extra_attribs


# In[44]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# In[45]:


#...........................................PIPELINING.................................................................

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([('selector', DataFrameSelector(num_attribs)),('imputer', SimpleImputer(strategy="median")),('attribs_adder', CombinedAttributesAdder()),('std_scaler', StandardScaler()),])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),('one_hot_encoder', OneHotEncoder()),])

full_pipeline = FeatureUnion(transformer_list=[("num_pipeline", num_pipeline),("cat_pipeline", cat_pipeline),])


# In[46]:


housing


# In[47]:


housing_prepared = full_pipeline.fit_transform(housing)


# In[48]:


housing_prepared


# In[49]:


housing_prepared.shape


# In[51]:


#....................................SELECTING AND TRAINING A MODEL.......................................

from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)


# In[52]:


#............................ROOT MEAN SQUARE ERROR........................................................

from sklearn.metrics import mean_squared_error
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)


# In[53]:


forest_rmse


# In[57]:


import warnings
warnings.simplefilter(action='ignore', category=Warning)


# In[55]:


#.............................................CROSS VALIDATION.....................................................

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[56]:


rmse_scores


# In[57]:


def display_scores(scores):
       print("Scores:", scores)
       print("Mean:", scores.mean())
       print("Standard deviation:", scores.std())


# In[58]:


display_scores(rmse_scores)


# In[59]:


#...............................FINE TUNING THE MODEL.................................................................

from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [10,30,40,50], 'max_features': [2, 4, 6, 8]},{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[60]:


grid_search.best_estimator_


# In[61]:


#.......................................EVALUATING TEST SYSTEM............................................................

final_model = grid_search.best_estimator_

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


# In[62]:


final_rmse


# In[63]:


final_model.score(X_test_prepared,y_test)


# In[58]:


from xgboost import XGBRegressor


# In[79]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'n_estimators': [10,60,1000], 'max_features': [2, 4, 6, 8]},{'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},]
xgbr = XGBRegressor()
grid_search = GridSearchCV(xgbr, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# In[75]:


grid_search.best_estimator_


# In[80]:


final_model = grid_search.best_estimator_

X_test = test_set.drop("median_house_value", axis=1)
y_test = test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)


# In[81]:


final_model.score(X_test_prepared,y_test)


# In[ ]:




