#!/usr/bin/env python
# coding: utf-8

# # **Assignment For Numpy**

# Difficulty Level **Beginner**

# 1. Import the numpy package under the name np

# In[1]:


import numpy as np


# 2. Create a null vector of size 10 

# In[2]:


null_vector=np.zeros(0)


# 3. Create a vector with values ranging from 10 to 49

# In[3]:


vector= np.arange(10,49)


# 4. Find the shape of previous array in question 3

# In[4]:


vector= np.arange(10,49)
np.shape(vector)


# 5. Print the type of the previous array in question 3

# In[5]:


vector= np.arange(10,49)
vector.dtype


# 6. Print the numpy version and the configuration
# 

# In[6]:


np.__version__


# In[7]:


np.show_config()


# 7. Print the dimension of the array in question 3
# 

# In[8]:


vector= np.arange(10,49)
vector.ndim


# 8. Create a boolean array with all the True values

# In[9]:


arr=np.ones(5,dtype= bool)
arr


# 9. Create a two dimensional array
# 
# 
# 

# In[10]:


array1=([[1,2,3],[4,5,6]])
array2=([[11,12,13],[14,15,16]])

np.concatenate((array1,array2),axis=0)


# 10. Create a three dimensional array
# 
# 

# In[11]:


a=np.arange(24).reshape(2,3,4)
a


# Difficulty Level **Easy**

# 11. Reverse a vector (first element becomes last)

# In[12]:


arr=np.arange(1,10)
arr


# 12. Create a null vector of size 10 but the fifth value which is 1 

# In[13]:


x = np.zeros(10)
x
x[5] = 1
x


# 13. Create a 3x3 identity matrix

# In[14]:


matrix=np.identity(3)
print(matrix)


# 14. arr = np.array([1, 2, 3, 4, 5]) 
# 
# ---
# 
#  Convert the data type of the given array from int to float 

# In[15]:


arr = np.array([1, 2, 3, 4, 5])
arr.astype(float)


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

# In[16]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]])  
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
arr1 * arr2


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

# In[17]:


arr1 = np.array([[1., 2., 3.],[4., 5., 6.]]) 
arr2 = np.array([[0., 4., 1.],[7., 2., 12.]])
comparison =arr1 == arr2
equal=comparison.all()
equal


# 17. Extract all odd numbers from arr with values(0-9)

# In[18]:


arr=np.arange(0,9)
arr[1:9:2]


# 18. Replace all odd numbers to -1 from previous array

# In[19]:


arr=np.arange(0,9)
for i in arr:
    if i % 2 >0:
        arr=-1
        print(arr)
    


# 19. arr = np.arange(10)
# 
# 
# ---
# 
# Replace the values of indexes 5,6,7 and 8 to **12**

# In[20]:


arr = np.arange(10)
arr[5]=12
arr[6]=12
arr[7]=12
arr[8]=12
for i in arr:
    print(i)


# 20. Create a 2d array with 1 on the border and 0 inside

# In[21]:


x=np.ones((4,3))
x
x[1:-1,1:-1]=0
x


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

# In[22]:


arr2d = np.array([[1, 2, 3],[4, 5, 6], [7, 8, 9]])
arr2d[1,1]=12
arr2d


# 22. arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
# 
# ---
# Convert all the values of 1st array to 64
# 

# In[24]:


arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
arr3d[0]=64
arr3d


# 23. Make a 2-Dimensional array with values 0-9 and slice out the first 1st 1-D array from it

# In[28]:


arr= np.arange(0,9).reshape(3,3)
arr[0]


# 24. Make a 2-Dimensional array with values 0-9 and slice out the 2nd value from 2nd 1-D array from it

# In[31]:


arr= np.arange(0,9)
arr.reshape(3,3)
arr[1]


# 25. Make a 2-Dimensional array with values 0-9 and slice out the third column but only the first two rows

# In[34]:


arr= np.arange(0,9).reshape(3,3)
a=arr[0]
b=arr[1]
print(a)
print(b)


# 26. Create a 10x10 array with random values and find the minimum and maximum values

# In[36]:


arr=np.random.randint(100,size=(10,10))
arr
print(np.min(arr))
print(np.max(arr))


# 27. a = np.array([1,2,3,2,3,4,3,4,5,6]) b = np.array([7,2,10,2,7,4,9,4,9,8])
# ---
# Find the common items between a and b
# 

# In[37]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
c=np.intersect1d(a,b)
c


# 28. a = np.array([1,2,3,2,3,4,3,4,5,6])
# b = np.array([7,2,10,2,7,4,9,4,9,8])
# 
# ---
# Find the positions where elements of a and b match
# 
# 

# In[59]:


a = np.array([1,2,3,2,3,4,3,4,5,6]) 
b = np.array([7,2,10,2,7,4,9,4,9,8])
np.where(a == b)


# 29.  names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])  data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will**
# 

# In[43]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[names != 'Will']


# 30. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe']) data = np.random.randn(7, 4)
# 
# ---
# Find all the values from array **data** where the values from array **names** are not equal to **Will** and **Joe**
# 
# 

# In[48]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.randn(7, 4)
data[names != 'Will']
data[names != 'Joe']


# Difficulty Level **Hard**

# 31. Create a 2D array of shape 5x3 to contain decimal numbers between 1 and 15.

# In[61]:


rand_arr = np.random.uniform(2,14, size=(5,3))
print(rand_arr)


# 32. Create an array of shape (2, 2, 4) with decimal numbers between 1 to 16.

# In[70]:


arr=np.random.randint(2,15,size=(2,2,4))
arr


# 33. Swap axes of the array you created in Question 32

# In[74]:


arr=np.random.randint(2,15,size=(2,2,4))
arr
np.swapaxes(arr,0,0)


# 34. Create an array of size 10, and find the square root of every element in the array, if the values less than 0.5, replace them with 0

# In[76]:


arr=np.arange(10)
arr=np.sqrt(arr)
np.where(arr<0.5,0,arr)


# 35. Create two random arrays of range 12 and make an array with the maximum values between each element of the two arrays

# In[77]:


arr1=np.random.randn(12)
arr2=np.random.randn(12)
np.maximum(a,b)


# 36. names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# 
# ---
# Find the unique names and sort them out!
# 

# In[78]:


names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
np.unique(names)


# 37. a = np.array([1,2,3,4,5])
# b = np.array([5,6,7,8,9])
# 
# ---
# From array a remove all items present in array b
# 
# 

# In[79]:


a = np.array([1,2,3,4,5]) 
b = np.array([5,6,7,8,9])
np.setdiff1d(a,b)


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

# In[109]:


sampleArray = np.array([[34,43,73],[82,22,12],[53,94,66]])
sampleArray[0,1]=10
sampleArray[1,1]=10
sampleArray[2,1]=10
sampleArray


# 39. x = np.array([[1., 2., 3.], [4., 5., 6.]]) y = np.array([[6., 23.], [-1, 7], [8, 9]])
# 
# 
# ---
# Find the dot product of the above two matrix
# 

# In[110]:


x = np.array([[1., 2., 3.], [4., 5., 6.]]) 
y = np.array([[6., 23.], [-1, 7], [8, 9]])
np.dot(a,b)


# 40. Generate a matrix of 20 random values and find its cumulative sum

# In[113]:


matrix= np.random.randint(20,size=(4,4))
matrix
np.sum(matrix)

