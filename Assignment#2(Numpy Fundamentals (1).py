#!/usr/bin/env python
# coding: utf-8

# # Numpy_Assignment_2::

# In[1]:


import numpy as np


# ## Question:1

# ### Convert a 1D array to a 2D array with 2 rows?

# #### Desired output::

# array([[0, 1, 2, 3, 4],
#         [5, 6, 7, 8, 9]])

# In[4]:


arr1=np.array([0, 1, 2, 3, 4,5, 6, 7, 8, 9])
arr2=np.reshape(arr1,(2,5))
arr2


# ## Question:2

# ###  How to stack two arrays vertically?

# #### Desired Output::
array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
# In[9]:


arr1=np.array([[0, 1, 2, 3, 4],
        [5, 6, 7, 8, 9]])
arr2=np.array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]])
np.vstack((arr1,arr2))


# ## Question:3

# ### How to stack two arrays horizontally?

# #### Desired Output::
array([[0, 1, 2, 3, 4, 1, 1, 1, 1, 1],
       [5, 6, 7, 8, 9, 1, 1, 1, 1, 1]])
# In[13]:


arr1=np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
arr2=np.array([[1,1,1,1,1], [1, 1, 1, 1, 1]])
np.hstack((arr1,arr2))              
              


# ## Question:4

# ### How to convert an array of arrays into a flat 1d array?

# #### Desired Output::
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# In[19]:


arr1=np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr2=arr1.flatten()
arr2


# ## Question:5

# ### How to Convert higher dimension into one dimension?

# #### Desired Output::
array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
# In[17]:


arr=np.arange(0,15).reshape(1,3,5)
print(arr.ndim,"Dimension")
print(arr)
arr=arr.flatten()
print(arr.ndim, "Dimension")
arr


# ## Question:6

# ### Convert one dimension to higher dimension?

# #### Desired Output::
array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
# In[21]:


arr1=np.array([[ 0, 1, 2],
[ 3, 4, 5],
[ 6, 7, 8],
[ 9, 10, 11],
[12, 13, 14]])
arr2=arr1.reshape(1,5,3)
arr2


# ## Question:7

# ### Create 5x5 an array and find the square of an array?

# In[28]:


arr1=np.arange(1,26).reshape(5,5)
print(arr1)
np.square(arr1)


# ## Question:8

# ### Create 5x6 an array and find the mean?

# In[35]:


arr1=np.arange(1,31).reshape(5,6)
np.mean(arr1)


# ## Question:9

# ### Find the standard deviation of the previous array in Q8?

# In[36]:


arr1=np.arange(1,31).reshape(5,6)
np.std(arr1)


# ## Question:10

# ### Find the median of the previous array in Q8?

# In[38]:


arr1=np.arange(1,31).reshape(5,6)
np.median(arr1)


# ## Question:11

# ### Find the transpose of the previous array in Q8?

# In[43]:


arr1=np.arange(1,31).reshape(5,6)
arr1.T


# ## Question:12

# ### Create a 4x4 an array and find the sum of diagonal elements?

# In[45]:


a=np.arange(1,17).reshape(4,4)
print("original matrx is :\n",a)
l=np.trace(a) 
print("trace of matrix: ",m)


# ## Question:13

# ### Find the determinant of the previous array in Q12?

# In[46]:


a=np.arange(1,17).reshape(4,4)
print(a)
det=np.linalg.det(a) 
print("Determinant of a is ",det)  


# ## Question:14

# ### Find the 5th and 95th percentile of an array?

# In[48]:


array1= np.array([3,4,6,2,10])
print("Array:",arr1)
print("5th percentile of an array:",np.percentile(arr1,5))
print("95th percentile of an array:",np.percentile(arr1,95))


# ## Question:15

# ### How to find if a given array has any null values?

# In[52]:


arr1=np.array([1,4,8,0])
arr2=np.sum(arr1)
np.isnan(arr2)


# In[ ]:




