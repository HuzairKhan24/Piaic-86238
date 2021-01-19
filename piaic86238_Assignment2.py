#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[4]:


import numpy as np
arr= np.arange(0,10).reshape(2,5)
arr


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[14]:


arr1= np.arange(10).reshape(2,5)
arr2= np.ones(10,dtype=int).reshape(2,5)
np.vstack((arr1,arr2))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[15]:


arr1= np.arange(10).reshape(2,5)
arr2= np.ones(10,dtype=int).reshape(2,5)
np.hstack((arr1,arr2))


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[18]:


arr=np.arange(0,10)
arr.flatten()


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[23]:


arr= np.arange(0,15).reshape(3,5)
print(arr)
arr=arr.flatten()
arr


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[24]:


arr= np.arange(0,15).reshape(5,3)
arr


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[26]:


arr= np.arange(25).reshape(5,5)
np.square(arr)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[30]:


arr= np.arange(0,30).reshape(5,6)
np.mean(arr)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[32]:


arr= np.arange(0,30).reshape(5,6)
np.std(arr)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[33]:


arr= np.arange(0,30).reshape(5,6)
np.median(arr)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[34]:


arr= np.arange(0,30).reshape(5,6)
np.transpose(arr)


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[49]:


arr=np.arange(0,16).reshape(4,4)
d=np.diagonal(arr)
result=np.sum(d)
print('Diagonal Elements',d)
print('result','=',result)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[50]:


arr=np.arange(0,16).reshape(4,4)
np.linalg.det(arr)


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[51]:


arr= np.arange(10)
np.percentile(arr,5)
np.percentile(arr,95)


# ## Question:15

# ### How to find if a given array has any null values?

# In[54]:


arr= np.arange(15).reshape(3,5)
arr
np.isnan(arr)

