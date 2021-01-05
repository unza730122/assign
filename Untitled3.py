#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# # 1 Method:rershape()

# In[2]:


arr=np.arange(1,11).reshape(2,5)
arr


# # 2 Method:array indexing

# In[3]:


arr=np.array([1,2,4,6])
arr[2]


# # 3 ndim Method

# In[4]:


array1d = np.array([1, 2, 3, 4, 5, 6])
array1d.ndim  
 


# # 4 resize() method

# In[8]:


arr = np.array([0,8,5,88,9,7, 8])
arr.resize(4)
arr


# # 5 reshape() Method

# In[11]:


arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8])
arr1 = arr1.reshape(2, 4)
arr1


# # 6 linspace() method

# In[12]:


arr2d = np.linspace(1, 12, 12).reshape(3, 4)
print(arr2d)


# # 7 longspace method()

# In[13]:


thearray = np.logspace(5, 10, num=10, base=10000000.0, dtype=float)
print(thearray)


# # 8 zeros() method

# In[15]:


arr = np.zeros((2, 10))
print(arr)


# # 9 ones() method

# In[16]:


array2d = np.ones((2, 4))
print(array2d)


# # 10 full() method

# In[17]:


array2d = np.full((2, 4), 3)
print(array2d)


# # eye method

# In[18]:


array1 = np.eye(3, dtype=int)
print(array1)
 


# # random rand()

# In[19]:


print(np.random.rand(3, 2))  # Uniformly distributed values.


# # random.randn()

# In[ ]:


print(np.random.randn(3, 2))  # Normally distributed values.


# # random.randint()

# In[ ]:


# Uniformly distributed integers in a given range.
print(np.random.randint(2, size=10))
print(np.random.randint(5, size=(2, 4)))


# # identity method

# In[20]:


print(np.identity(3))
 
print(np.diag(np.arange(0, 8, 2)))
 
print(np.diag(np.diag(np.arange(9).reshape((3,3)))))


# # Transpose method

# In[23]:


array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array2d=array2d.T
array2d


# # Rotation method

# In[26]:


array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
arrayRot90 = np.rot90(array2d)
arrayRot90


# # fliplr method

# In[27]:


array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Flip array in the left/right direction.
arrayFlr = np.fliplr(array2d)
print(arrayFlr)


# # flipud method

# In[28]:


array2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Flip array in the up/down direction.
arrayFud = np.flipud(array2d)
print(arrayFud)


# # hstack method

# In[30]:


array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])
 
# Stack arrays in sequence horizontally (column wise).
arrayH = np.hstack((array1, array2))
print(arrayH)
 


# In[ ]:


#vstack method


# In[31]:


array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]]) 
# Stack arrays in sequence vertically (row wise).
arrayV = np.vstack((array1, array2))
print(arrayV)
 


# In[ ]:


#dstack method


# In[32]:


array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]]) 
# Stack arrays in sequence depth wise (along third axis).
arrayD = np.dstack((array1, array2))
print(arrayD)
 


# In[ ]:


#concatenate  method


# In[33]:


array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]]) 
# Appending arrays after each other, along a given axis.
arrayC = np.concatenate((array1, array2))
print(arrayC)
 


# In[ ]:


#append method


# In[34]:


array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]]) 
# Append values to the end of an array.
arrayA = np.append(array1, array2, axis=0)
print(arrayA)
 


# In[29]:


array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])
arrayA = np.append(array1, array2, axis=1)
print(arrayA)
 


# In[ ]:


#where method


# In[36]:


arr1 = np.array([[1, 2, 3], [4, 5, 6]])
 
arr2 = np.where(arr1 < 4, arr1 * 2, arr1 * 3)
 
print(arr2)


# In[ ]:


#mean method


# In[39]:


array1 = np.array([[10, 20, 30], [40, 50, 60]])
print("Mean: ", np.mean(array1))


# In[ ]:


standard method


# In[40]:


array1 = np.array([[10, 20, 30], [40, 50, 60]])
print("Std: ", np.std(array1))


# In[ ]:


#variance method


# In[41]:


array1 = np.array([[10, 20, 30], [40, 50, 60]])
print("Var: ", np.var(array1))


# In[ ]:


#sum method


# In[43]:


array1 = np.array([[10, 20, 30], [40, 50, 60]])
print("Sum: ", np.sum(array1)) 


# In[ ]:


prod method


# In[42]:


array1 = np.array([[10, 20, 30], [40, 50, 60]])
print("Prod: ", np.prod(array1))


# In[ ]:


#split method


# In[44]:


arr = np.array([1, 2, 3, 4, 5, 6])
newarr = np.array_split(arr, 3)
print(newarr)


# In[ ]:


#sort method


# In[45]:


arr = np.array([3, 2, 0, 1])
print(np.sort(arr))


# In[ ]:


#dot method


# In[46]:


a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
np.dot(a,b)


# In[ ]:


#vdot method


# In[48]:


a = np.array([[1,2],[3,4]]) 
b = np.array([[11,12],[13,14]]) 
print(np.vdot(a,b))


# In[ ]:


#determinant method


# In[50]:


a = np.array([[1,2], [3,4]]) 
print(np.linalg.det(a))


# In[ ]:


#inverse method


# In[54]:


x = np.array([[1,2],[3,4]]) 
y = np.linalg.inv(x) 
print(x) 
print(y) 
print(np.dot(x,y))


# In[ ]:


#intersect1d() method


# In[56]:


rr1 = np.array([1, 1, 2, 3, 4]) 
arr2 = np.array([2, 1, 4, 6]) 
    
gfg = np.intersect1d(arr1, arr2) 
    
print (gfg) 


# In[ ]:


#union1d() method


# In[57]:


arr1 = [-1, 0, 1] 
arr2 = [-2, 0, 2] 
   
gfg = np.union1d(arr1, arr2) 
   
print (gfg) 


# In[ ]:


#unique() method


# In[61]:


x = np.array([[1, 1], [2,3], [3,4]])
np.unique(x)


# In[ ]:


#in1d() method


# In[62]:


arr1 = np.array([0, 1, 2, 3, 0, 4, 5]) 
arr2 = [0, 2, 5] 
gfg = np.in1d(arr1, arr2) 
print (gfg) 


# In[ ]:


#setdiff1d() method


# In[63]:


arr1 = [5, 6, 2, 3, 4] 
arr2 = [1, 2, 3] 
gfg = np.setdiff1d(arr1, arr2) 
print (gfg) 


# In[ ]:


#setxor1d() method


# In[64]:


arr1 = [1, 2, 3, 4] 
arr2 = [2, 4, 6, 8] 
gfg = np.setxor1d(arr1, arr2) 
print (gfg)


# In[ ]:


#max() method


# In[67]:


arr = np.array([1, 5, 4, 8, 3, 7])  
max_element = np.max(arr) 
print('maximum element in the array is: ',  
      max_element) 


# In[ ]:


#minimum method


# In[68]:


arr = np.array([1, 5, 4, 8, 3, 7]) 
min_element = np.min(arr)
print('minimum element in the array is: ', 
      min_element)


# In[ ]:


#argmax() method


# In[70]:


array = np.arange(12).reshape(3, 4) 
print("INPUT ARRAY : \n", array) 
print("\nMax element : ", np.argmax(array)) 
print("\nIndices of Max element : ", np.argmax(array, axis=0)) 
print("\nIndices of Max element : ", np.argmax(array, axis=1)) 


# In[ ]:


#argmin() method


# In[72]:


array = np.arange(12).reshape(3, 4) 
print("INPUT ARRAY : \n", array) 
print("\nMin element : ", np.argmin(array)) 
print("\nIndices of Min element : ", np.argmin(array, axis=0)) 
print("\nIndices of Min element : ", np.argmin(array, axis=1)) 


# In[ ]:


#cumsum method


# In[73]:


in_num = 10
  
print ("Input  number : ", in_num) 
    
out_sum = np.cumsum(in_num)  
print ("cumulative sum of input number : ", out_sum)  

