#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# In[57]:


from textblob import TextBlob
food = pd.read_csv("C:\\Users\\rsury\\Flashmarket\\food.csv")


# In[58]:


food.head()


# In[59]:


food.shape


# In[60]:


polarity_score = []

for i in range(0,food.shape[0] ):
    score = TextBlob(food.iloc[i][0])
    score1 = score.sentiment[0]
    polarity_score.append(score1)


# In[61]:


j = list(polarity_score)


# In[62]:


food = pd.concat([food,pd.Series(j)] , axis =1 )


# In[63]:


food.head()


# In[64]:


food.rename(columns={food.columns[1] :"sentiment"}, inplace = True)


# In[65]:


food.head()


# In[66]:


from wordcloud import WordCloud
WordCloud


# In[67]:


nltk.download('stopwords')


# In[68]:


cloud = WordCloud(max_words= 50, stopwords=stopwords.words("english") ).generate(str(food['good']) )
plt.figure(figsize= (20,20))
plt.imshow(cloud)

