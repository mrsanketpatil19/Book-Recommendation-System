#!/usr/bin/env python
# coding: utf-8

# In[36]:


# Importing required libraries
import numpy as np
import pandas as pd


# In[37]:


pwd


# In[38]:


books = pd.read_csv(r'C:\Users\Sanket Patil\Documents\archive (2)\Books.csv')
users = pd.read_csv(r'C:\Users\Sanket Patil\Documents\archive (2)\Users.csv')
ratings = pd.read_csv(r'C:\Users\Sanket Patil\Documents\archive (2)\Ratings.csv')


# In[39]:


books.head()


# In[40]:


users.head()


# In[41]:


ratings.head()


# In[42]:


print(books.shape)
print(users.shape)
print(ratings.shape)


# In[43]:


books.isnull().sum()


# In[44]:


users.isnull().sum()


# In[45]:


ratings.isnull().sum()


# In[46]:


# checking if data has duplicate values
books.duplicated().sum()
users.duplicated().sum()
ratings.duplicated().sum()


# ## Popularity based recommender system

# In[47]:


ratings_with_name = ratings.merge(books,on='ISBN')
ratings_with_name.shape
ratings_with_name.head()
# If you look at the shape of the new dataframe, rows reduced because there are some books which are in ratings but not in books.


# In[60]:


# Finding book names and count of number of ratings on that respective book. 
num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns = {'Book-Rating' : 'num_rating'},inplace=True)
num_rating_df


# In[54]:


ratings_with_name['Book-Rating'].dtype
ratings_with_name['Book-Rating'].mean()


# In[61]:


# Finding book names and count of number of ratings on that respective book. 
avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns = {'Book-Rating' : 'avg_rating'},inplace=True)
avg_rating_df



# In[64]:


popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
popular_df


# In[68]:


popular_df = popular_df[popular_df['num_rating']>=250].sort_values('avg_rating', ascending=False).head(50)


# In[74]:


popular_df = popular_df.merge(books, on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_rating','avg_rating']]


# Collaborative Filtering Based Recommender System

# In[82]:


x = ratings_with_name.groupby('User-ID').count()['Book-Rating']>200
padhe_likhe_users = x[x].index


# In[86]:


# Filtering only those users which have given a minimum 200 ratings so that we can say these users are well educated in reading books
filtered_rating = ratings_with_name[ratings_with_name['User-ID'].isin(padhe_likhe_users)]


# In[94]:


y = filtered_rating.groupby('Book-Title').count()['Book-Rating']>=50
famous_books = y[y].index


# In[100]:


# Filtering those books who have got ratings greater than 50
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[103]:


pt = final_ratings.pivot_table(index= 'Book-Title', columns='User-ID',values='Book-Rating')
pt


# In[105]:


pt.fillna(0, inplace=True)


# In[106]:


pt


# In[108]:


from sklearn.metrics.pairwise import cosine_similarity


# In[110]:


cosine_similarity(pt).shape


# In[111]:


# Calculating the euclidian distance of each book in a multidimentional space with every other book
similarity_scores = cosine_similarity(pt)


# In[122]:


similarity_scores


# In[150]:


def recommend(book_name):
    #index fetch
    index = np.where(pt.index==book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    for i in similar_items:
        print(pt.index[i[0]])


# In[157]:


recommend('A Bend in the Road')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




