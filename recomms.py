#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv(r"/Users/nikhildonde/Downloads/Project 2  - Spotify Songs’ Genre Segmentation/spotify dataset.csv")
df


# In[4]:


df['ID'] = df.index
df


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


print(pd.unique(df['playlist_subgenre']))


# In[8]:


df.isnull().sum()


# In[9]:


df=df.dropna()
df


# In[10]:


def categorical_histogram(df, playlist_genre):
    plt.barh(df[playlist_genre].value_counts().index, df[playlist_genre].value_counts().values)
    plt.xlabel('Count')
    plt.ylabel('playlist genre')
    plt.title('Categorical Histogram of Playlist Genre')
    plt.show()
    
column_name='playlist_genre'
categorical_histogram(df, "playlist_genre")


# In[11]:


def value_plot(df, track_popularity):
    sns.lineplot(data=df,x=df.index,y=track_popularity,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('track popularity')
    plt.title('Graph of track popularity')
    plt.show()
    
column_name = 'Track Popularity'
value_plot(df, "track_popularity")

def value_plot(df, danceability):
    sns.lineplot(data=df,x=df.index,y=danceability,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('danceability')
    plt.title('Graph of danceability')
    plt.show()
    
column_name = 'danceability'
value_plot(df, "danceability")

def value_plot(df, energy):
    sns.lineplot(data=df,x=df.index,y=energy,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('Energy')
    plt.title('Graph of energy')
    plt.show()
    
column_name = 'energy'
value_plot(df, "energy")

def value_plot(df, key):
    sns.lineplot(data=df,x=df.index,y=key,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('key')
    plt.title('Graph of key')
    plt.show()
    
column_name = 'key'
value_plot(df, "key")

def value_plot(df, loudness):
    sns.lineplot(data=df,x=df.index,y=loudness,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('loudness')
    plt.title('Graph of loudness')
    plt.show()
    
column_name = ('loudness')
value_plot(df, "loudness")

def value_plot(df, mode):
    sns.lineplot(data=df,x=df.index,y=mode,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('mode')
    plt.title('Graph of mode')
    plt.show()
    
column_name = ('mode')
value_plot(df, "mode")

def value_plot(df, speechiness):
    sns.lineplot(data=df,x=df.index,y=speechiness,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('speechiness')
    plt.title('Graph of speechiness')
    plt.show()
    
column_name = ('speechiness')
value_plot(df, "speechiness")

def value_plot(df, acousticness):
    sns.lineplot(data=df,x=df.index,y=acousticness,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('acousticness')
    plt.title('Graph of acousticness')
    plt.show()
    
column_name = ('acousticness')
value_plot(df, "acousticness")

def value_plot(df, instrumentalness):
    sns.lineplot(data=df,x=df.index,y=instrumentalness,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('instrumentalness')
    plt.title('Graph of instrumentalness')
    plt.show()
    
column_name = ('instrumentalness')
value_plot(df, "instrumentalness")

def value_plot(df, liveness):
    sns.lineplot(data=df,x=df.index,y=liveness,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('liveness')
    plt.title('Graph of liveness')
    plt.show()
    
column_name = ('liveness')
value_plot(df, "liveness")

def value_plot(df, valence):
    sns.lineplot(data=df,x=df.index,y=valence,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('valence')
    plt.title('Graph of valence')
    plt.show()
    
column_name = ('valence')
value_plot(df, "valence")

def value_plot(df, tempo):
    sns.lineplot(data=df,x=df.index,y=tempo,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('tempo')
    plt.title('Graph of tempo')
    plt.show()
    
column_name = ('tempo')
value_plot(df, "tempo")

def value_plot(df, duration_ms):
    sns.lineplot(data=df,x=df.index,y=duration_ms,hue='playlist_genre',palette='rainbow')
    plt.xlabel('Index')
    plt.ylabel('duration_ms')
    plt.title('Graph of duration_ms')
    plt.show()
    
column_name = ('duration_ms')
value_plot(df, "duration_ms")


# In[12]:


def histogram(df, track_popularity):
    plt.hist(df[track_popularity])
    plt.xlabel('track popularity')
    plt.ylabel('Count')
    plt.title('Histogram of track popularity')
    plt.show()

column_name = ('track popularity')
histogram(df, "track_popularity")

def histogram(df, danceability):
    plt.hist(df[danceability])
    plt.xlabel('danceability')
    plt.ylabel('Count')
    plt.title('Histogram of danceability')
    plt.show()

column_name = ('danceability')
histogram(df, "danceability")

def histogram(df, energy):
    plt.hist(df[energy])
    plt.xlabel('energy')
    plt.ylabel('Count')
    plt.title('Histogram of energy')
    plt.show()

column_name = ('energy')
histogram(df, "energy")

def histogram(df, key):
    plt.hist(df[key])
    plt.xlabel('key')
    plt.ylabel('Count')
    plt.title('Histogram of key')
    plt.show()

column_name = ('key')
histogram(df, "key")

def histogram(df, loudness):
    plt.hist(df[loudness])
    plt.xlabel('loudness')
    plt.ylabel('Count')
    plt.title('Histogram of loudness')
    plt.show()

column_name = ('loudness')
histogram(df, "loudness")

def histogram(df, mode):
    plt.hist(df[mode])
    plt.xlabel('mode')
    plt.ylabel('Count')
    plt.title('Histogram of mode')
    plt.show()

column_name = ('mode')
histogram(df, "mode")

def histogram(df, speechiness):
    plt.hist(df[speechiness])
    plt.xlabel('speechiness')
    plt.ylabel('Count')
    plt.title('Histogram of speechiness')
    plt.show()

column_name = ('speechiness')
histogram(df, "speechiness")

def histogram(df, acousticness):
    plt.hist(df[acousticness])
    plt.xlabel('acousticness')
    plt.ylabel('Count')
    plt.title('Histogram of acousticness')
    plt.show()

column_name = ('acousticness')
histogram(df, "acousticness")

def histogram(df, instrumentalness):
    plt.hist(df[instrumentalness])
    plt.xlabel('instrumentalness')
    plt.ylabel('Count')
    plt.title('Histogram of instrumentalness')
    plt.show()

column_name = ('instrumentalness')
histogram(df, "instrumentalness")

def histogram(df, liveness):
    plt.hist(df[liveness])
    plt.xlabel('liveness')
    plt.ylabel('Count')
    plt.title('Histogram of liveness')
    plt.show()

column_name = ('liveness')
histogram(df, "liveness")

def histogram(df, valence):
    plt.hist(df[valence])
    plt.xlabel('valence')
    plt.ylabel('Count')
    plt.title('Histogram of valence')
    plt.show()

column_name = ('valence')
histogram(df, "valence")

def histogram(df, tempo):
    plt.hist(df[tempo])
    plt.xlabel('tempo')
    plt.ylabel('Count')
    plt.title('Histogram of tempo')
    plt.show()

column_name = ('tempo')
histogram(df, "tempo")

def histogram(df, duration_ms):
    plt.hist(df[duration_ms])
    plt.xlabel('duration ms')
    plt.ylabel('Count')
    plt.title('Histogram of duration ms')
    plt.show()

column_name = ('duration_ms')
histogram(df, "duration_ms")


# In[13]:


plt.scatter(df.index,df.danceability)


# In[14]:


plt.scatter(df.index,df.energy)


# In[15]:


plt.scatter(df.index,df.energy)


# In[16]:


plt.scatter(df.index,df.key)


# In[17]:


plt.scatter(df.index,df.loudness)


# In[18]:


plt.scatter(df.index,df.speechiness)


# In[19]:


plt.scatter(df.index,df.acousticness)


# In[20]:


plt.scatter(df.index,df.instrumentalness)


# In[21]:


plt.scatter(df.index,df.liveness)


# In[22]:


plt.scatter(df.index,df.valence)


# In[23]:


plt.scatter(df.index,df.tempo)


# In[24]:


plt.scatter(df.index,df.duration_ms)


# In[25]:


y=df.iloc[:,11:23]
y


# In[26]:


extracted_col = df["track_popularity"]
display(extracted_col)
  
y.insert(10, "track_popularity", extracted_col)
display(y)


# In[35]:


fig,axes=plt.subplots()

axes.scatter(df.index,df.danceability)
axes.scatter(df.index,df.energy)
axes.scatter(df.index,df.key)
axes.scatter(df.index,df.loudness)
axes.scatter(df.index,df.acousticness)
axes.scatter(df.index,df.instrumentalness)
axes.scatter(df.index,df.liveness)
axes.scatter(df.index,df.valence)
axes.scatter(df.index,df.tempo)
axes.scatter(df.index,df.duration_ms)
axes.scatter(df.index,df.track_popularity)

plt.title('Combined Cluster Graph')
plt.xlabel('Index')
plt.ylabel('Parameters')
plt.show()


# In[43]:


from sklearn.cluster import KMeans
km=KMeans(n_clusters=24)
km.fit(y)


# In[44]:


df['cluster group no.']=km.labels_
df


# In[45]:


df['cluster group no.'].value_counts()


# In[46]:


sns.scatterplot(data=y)


# In[47]:


km.cluster_centers_ 


# In[ ]:




