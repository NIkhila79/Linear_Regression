
# coding: utf-8

# In[133]:


import os


# In[134]:


os.chdir("E:\\spider_exer\\30_9_18")


# In[135]:


ls


# In[154]:


import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


# In[155]:


df = pd.read_csv("train.csv")


# In[156]:


df.shape


# In[157]:


correlation_values = df.select_dtypes(include=[np.number]).corr()
correlation_values


# In[158]:


correlation_values[["SalePrice"]]


# In[159]:


sf= correlation_values.loc[(correlation_values["SalePrice"]>=0.6) | (correlation_values["SalePrice"]<= -0.6)]
# selected_features = correlation_values[["SalePrice"]][(correlation_values["SalePrice"]>=0.6)|(correlation_values["SalePrice"]<=-0.6)]


# In[160]:


sf
# selected_features


# In[161]:


sf[["OverallQual","TotalBsmtSF","1stFlrSF","GrLivArea","GarageCars","GarageArea"]]


# In[162]:


X = df[["OverallQual","TotalBsmtSF","GarageCars","GarageArea"]]


# In[163]:


y = df["SalePrice"]


# In[164]:


from sklearn.model_selection import train_test_split as tts


# In[165]:


X_train, X_test, y_train, y_test = tts(X,y,test_size=0.3,random_state = 42)


# In[166]:


from sklearn.linear_model import LinearRegression


# In[167]:


reg= LinearRegression()


# In[168]:


reg.fit(X_train, y_train)


# In[169]:


y_pred = reg.predict(X_test)


# In[170]:


reg.score(X_test,y_test)


# In[173]:


rmse=np.sqrt(mean_squared_error(y_test, y_pred))
rmse

