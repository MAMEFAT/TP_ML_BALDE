#!/usr/bin/env python
# coding: utf-8

# ## Import des librairies 

# In[1]:


import pandas as mfb
import numpy as np
import pickle
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as mfb_tts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
from sklearn.pipeline import Pipeline
 


# ## Chargement des données

# In[2]:


positif = mfb.read_pickle('imdb_raw_pos.pickle')


# In[3]:


negatif = mfb.read_pickle('imdb_raw_neg.pickle')


# In[4]:


positif


# In[5]:


negatif


# In[6]:


negatif = {'review': negatif}


# In[7]:


df_negatif = mfb.DataFrame(negatif)


# In[8]:


df_negatif.shape


# In[9]:


df_negatif


# In[10]:


positif = {"review": positif}


# In[13]:


df_positif = mfb.DataFrame(positif)


# In[14]:


df_positif


# ## Rennomons les sentiments positifs et négatifs par 1 et 0

# Nous allons séparer et nommer les sentiments qui sont positifs par 1 et ceux qui sont négatifs par 0.

# In[15]:


df_positif["sentiment"]= 1


# In[16]:


df_negatif["sentiment"]= 0


# In[17]:


df_negatif = mfb.DataFrame(neg)


# In[18]:


df_negatif


# In[19]:


df_negatif.loc[df_negatif["review"]=='review' "sentiment"]=0
df_positif.loc[df_positif["review"]=='review' "sentiment"]=1


# In[20]:


df_positif


# In[21]:


df_negatif


# ## Concaténation des sentiments

# Nous allons maintenant rassembler les commentaires positifs et négatifs dans un même vecteur V.

# In[22]:


V=mfb.concat([df_positif,df_negatif])


# In[23]:


V


# ## Séparons les reviews des sentiments

# Nous allons répartir les données dans deux vecteurs, nous mettrons les sentiments dans le vecteur Y et les reviews dans le vecteur X.

# In[24]:


vect_X = V["review"]
vect_Y = V["sentiment"]


# In[25]:


vect_X


# In[26]:


vect_Y


# In[27]:


X_train, X_test, y_train, y_test = mfb_tts(vect_X,vect_Y, test_size =0.30, random_state =42) 


# In[28]:


X_train.shape


# ## Vectorisation

# In[29]:


CV = CountVectorizer(stop_words='english', binary=False, ngram_range=(1,1))


# In[32]:


X_traincv = CV.fit_transform(X_train)


# In[40]:


X_testcv = CV.transform(X_test)


# In[41]:


CV.vocabulary_


# ## Régression Logistique

# In[56]:


from sklearn.linear_model import LogisticRegression # modèele linèaire 
classificateur_imdb = LogisticRegression()
model=Pipeline([('vectorizer',tf),('classifier',classificateur_imdb)])


# In[57]:


reg = LogisticRegression(max_iter=50000) 


# In[58]:


X_traincv.shape


# In[59]:


y_train.shape


# In[60]:


reg_test = reg.fit(X_traincv,y_train)


# In[61]:


reg_test.score(X_traincv, y_train)


# In[62]:


X_testcv.shape


# In[63]:


y_test.shape


# In[64]:


reg_test2 = reg.fit(X_testcv, y_test)


# In[65]:


reg_test.score(X_testcv, y_test)


# In[66]:


model.fit(X_train,y_train) # apprentissage 


# In[67]:


ypred=model.predict(X_test)  # prediction 


# In[68]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
accuracy_score(ypred,y_test)
    


# In[69]:


matrice=confusion_matrix(y_test,ypred,labels=[1,0])
disp = ConfusionMatrixDisplay(confusion_matrix=matrice, display_labels=['Positif', 'Negatif'])
disp.plot()


# In[ ]:





# In[ ]:





# In[ ]:




