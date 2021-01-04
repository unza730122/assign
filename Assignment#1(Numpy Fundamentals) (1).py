#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[3]:


import numpy as np


# 2. Create a null vector of size 10 

# In[8]:


arr=np.zeros(10)


# 3. Create a vector with values ranging from 10 to 49

# In[10]:


ar1 = np.arange(10,50)


# 4. Find the shape of previous array in question 3

# In[11]:


ar1.shape


# 5. Print the type of the previous array in question 3

# In[12]:


ar1.dtype


# 6. Print the numpy version and the configuration
# 

# In[15]:


np.__version__  


# In[16]:


np.show_config()


# 7. Print the dimension of the array in question 3
# 

# In[17]:


ar1.ndim


# 8. Create a boolean array with all the True values

# In[18]:


ar3 = np.array([1,2,3,4,5,6,7,8,9,20])


# In[19]:


ar3>5


# 9. Create a two dimensional array
# 
# 
# 

# In[26]:


ar4 = np.arange(1,21).reshape(2,10)


# In[27]:


ar4


# 10. Create a three dimensional array
# 
# 

# In[28]:


ar4 = np.arange(1,22).reshape(3,7)


# In[29]:


ar4


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[32]:


s = np.arange(1,11)


# In[40]:


s = s[::-1]
s


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[39]:


d=np.zeros(10)
d[4]=1
d


# 13. Create a 3x3 identity matrix

# In[41]:


np.identity(3)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[44]:


arr= arr.astype('float64')
arr


# 15. arr1 =          np.array([[1., 2., 3.],
# 
#                     [4., 5., 6.]])  
#                       
#     arr2 = np.array([[0., 4., 1.],
#      
#                    [7., 2., 12.]])
# 
# ---
# 
# 
# Multiply arr1 with arr2
# 

# In[46]:


arr1 = np.array([[1., 2., 3.],

            [4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],

           [7., 2., 12.]])
arr1*arr2


# 16. arr1 = np.array([[1., 2., 3.],
#                     [4., 5., 6.]]) 
#                     
#     arr2 = np.array([[0., 4., 1.], 
#                     [7., 2., 12.]])
# 
# 
# ---
# 
# Make an array by comparing both the arrays provided above

# In[47]:


compare_array= arr1==arr2
compare_array


# 17. Extract all odd numbers from arr with values(0-9)

# In[51]:


array = np.array([0,1,2,3,4,5,6,7,8,9])
array[array%2==1]


# 18. Replace all odd numbers to -1 from previous array

# In[53]:


array = np.array([0,1,2,3,4,5,6,7,8,9])
array[array%2==1]=-1
array


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[56]:


arr = np.arange(10)
arr[4:7]=12
arr


# 20. Create a 2d array with 1 on the border and 0 inside

# In[75]:


arr = np.ones((5,5))
arr[1:-1,1:-1]=0
arr


# Difficulty Level **Medium**

# 21. arr2d = np.array([[1, 2, 3],
# 
#                     [4, 5, 6], 
# 
#                     [7, 8, 9]])
# 
# ---
# 
# Replace the value 5 to 12

# In[80]:


arr2d = np.array([[1, 2, 3],

            [4, 5, 6], 

            [7, 8, 9]])
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[84]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0,0:1]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[86]:


arr2d=np.array([[1,2,3,4,5],[6,7,8,9,10]])
arr2d[0,::]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[94]:


arr2d=np.array([[0,1,2,3,4],[5,6,7,8,9]])
arr2d[1, 1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[51]:


arrr2d = np.array([[0,1,2],[3,4,5],[6,7,8]])
arrr2d[:2,2]


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[54]:


x = np.random.randint(100, size=(10, 10))
x
x.max()
x.min()


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[58]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.intersect1d(a, b)


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[88]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a==b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[78]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
names
data
data[names != 'Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[87]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) 
data = np.random.randn(7, 4)
check = (names != 'Bob') & (names != 'Will')
data[check]


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[92]:


arrrr2=np.random.uniform(5,10 ,size=(5,3))
arrrr2


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[102]:


a=np.arange(1.,17.).reshape(2,2,4)
a


# 33. Swap axes of the array you created in Question 32

# In[103]:


a.swapaxes(1,2)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[109]:


a = np.arange(1,11)
a[a<0.5]=0
np.sqrt(a)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[20]:


a = np.random.randint(12)
b = np.random.randint(12)
arr=np.where(a>b , a ,b )


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[14]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# In[15]:


np.sort(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[18]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
result = np.setdiff1d(a,b)
result


# 38.  Following is the input NumPy array delete column two and insert following new column in its place.
# 
# ---
# sampleArray = numpy.array([[34,43,73],[82,22,12],[53,94,66]]) 
# 
# 
# ---
# 
# newColumn = numpy.array([[10,10,10]])
# 

# In[23]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray=np.delete(sampleArray, 1, axis=1)

newColumn = np.array([[10,10,10]])
sampleArray=np.insert(sampleArray, 1, newColumn,axis=1)
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[28]:


x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(x,y)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[32]:


matrixx=np.random.rand(20)
matrixx
cum=np.cumsum(matrixx)
cum


# In[ ]:




