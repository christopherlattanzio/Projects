
# coding: utf-8

# In[8]:

import pandas as pd
from pandas import Series , DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# In[9]:

titanic_df = pd.read_csv('train.csv')


# In[10]:

titanic_df.head()


# In[11]:

titanic_df.tail()


# In[12]:

titanic_df.info()


# In[23]:

sns.countplot('Sex',data=titanic_df)


# In[24]:

sns.countplot('Sex',data=titanic_df,hue='Pclass')


# In[25]:

sns.countplot('Pclass',data=titanic_df,hue='Sex')


# In[26]:

def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex


# In[28]:

titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child,axis=1)


# In[29]:

titanic_df[0:10]


# In[31]:

sns.countplot('Pclass',data=titanic_df,hue='person')


# In[33]:

titanic_df['Age'].hist(bins=70)


# In[34]:

titanic_df['Age'].mean()


# In[36]:

titanic_df['person'].value_counts()


# In[37]:

fig = sns.FacetGrid(titanic_df,hue='Sex',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[38]:

fig = sns.FacetGrid(titanic_df,hue='person',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[39]:

fig = sns.FacetGrid(titanic_df,hue='Pclass',aspect=4)

fig.map(sns.kdeplot,'Age',shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))

fig.add_legend()


# In[40]:

titanic_df.head()


# In[41]:

deck = titanic_df['Cabin'].dropna()


# In[42]:

deck.head()


# In[44]:

levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.countplot('Cabin',data=cabin_df,palette='winter_d')


# In[57]:

cabin_df = cabin_df[cabin_df.Cabin !='T']
sns.countplot('Cabin',data=cabin_df,palette='winter_d',order=['A','B','C','D','E','F'])


# In[47]:

titanic_df.head()


# In[56]:

sns.countplot('Embarked',data=titanic_df,hue='Pclass',order=['Q','C','S'])


# In[58]:

#family or alone?
titanic_df.head()


# In[60]:

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch


# In[61]:

titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone']==0] = 'Alone'


# In[62]:

titanic_df.head()


# In[65]:

sns.countplot('Alone',data=titanic_df,hue='Sex')


# In[66]:

titanic_df['Survivor'] = titanic_df.Survived.map({0:'no',1:'yes'})


# In[74]:

sns.countplot('Pclass',data=titanic_df,hue='Sex')


# In[76]:

sns.factorplot(x="Pclass", y="Survived", hue="person", data=titanic_df)


# In[77]:

sns.lmplot('Age','Survived',data=titanic_df)


# In[79]:

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter')


# In[81]:

generations = [10,20,30,40,60,70,80]

sns.lmplot('Age','Survived',hue='Pclass',data=titanic_df,palette='winter',x_bins=generations)


# In[85]:

sns.lmplot('Age','Survived',hue='Sex',data=titanic_df,palette='winter',x_bins=generations)


# In[ ]:



