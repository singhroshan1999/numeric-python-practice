#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
arr = [12,23,34,45,56,67,78,89,90]  # python list
C = np.array(arr)  # creating ndarray out of python list
print(arr)
print(C)
print(C**2)  # performing simple scalar multiplication
print(type(C))


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.plot(C)  # simple graph plotting
plt.show()


# ## Creatig numpy array

# In[10]:


# np.arange([start,][stop,][step,][dtype=type])  --> ndarray
a1 = np.arange(10)
print(a1,type(a1))

a2 = np.arange(10,50)
print(a2)

a3 = np.arange(0,10.4,0.7)
print(a3)

a4 = np.arange(10,dtype = float)
print(a4)


# In[23]:


# np.linspace(start, stop, num=50, endpoint=True, retstep=False)
# if endpoint == True spacing = (stop-start)/49 else (stop-start)/50

print(np.linspace(0,10))
print(np.linspace(5,15,num = 5))
print(np.linspace(3,9,5,endpoint = True))
print(np.linspace(3,9,5,endpoint = False))
print(np.linspace(100,108,9,True,retstep = True))
arr,step = np.linspace(100,108,9,True,retstep = True)
print(arr,step)


# In[57]:


# creating array of zeros, ones, empty

#  np.[zeros,ones,empty](dim_tuple,dtype)

x = np.zeros((2,3))
print(x)
y = np.ones((2,3),dtype = int)
print(y)
z = np.empty((2,3))
print(z)

# creating zeros, ones using existing shape

A = np.arange(50).reshape(5,10)
print(A)

a = np.ones_like(A)
b = np.zeros_like(A)
print(a)
print(b)


# In[59]:


# creating identity array

# using np.identity(n,dtype)

i1 = np.identity(5,dtype = int)
print(i1)

# using np.eye(N,M=none,k=0,dtype)

i2 = np.eye(4,5,k=-1,dtype = int)
i3 = np.eye(4,5,k=0,dtype = int)
print(i2,"\n",i3)


# ## method and function of ndarray

# In[25]:


# dimension of array np.ndim(array) --> dimenstion

a0 = np.array(42)
a1 = np.array([1,2,3,4,5])
a2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(np.ndim(a0),np.ndim(a1),np.ndim(a2))


# In[27]:


# getting type of array arr.dtype

ai = np.array([1,2,3,4])
af = np.array([1.5,2,3])
ac = np.array(["roshan","singh","eqwe"])
print(ai.dtype,af.dtype,ac.dtype)


# In[30]:


# multidimensional array using np.array()

A = np.array([ [3.4, 8.7, 9.9], 
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])
print(A)
print(A.ndim)

B = np.array([ [[111, 112], [121, 122]],
               [[211, 212], [221, 222]],
               [[311, 312], [321, 322]] ])
print(B)
print(B.ndim)

C = np.array([ [[111, 112], [121, 122]],
               [[211, 212], [221, 222]] ])
print(C)
print(C.ndim)


# In[33]:


# getting shape (axis element) of array np.shape(arr) or evuivalent arr.shape

x = np.array([ [67, 63, 87],
               [77, 69, 59],
               [85, 87, 99],
               [79, 72, 71],
               [63, 89, 93],
               [68, 92, 78]])
print(np.shape(x))
print(x.shape)

y = np.array(42)  # shape of 0d array
print(y.shape)

# changing shape of array

print(x)
x.shape = (3,6)
print(x)
x.shape = (2,9)
print(x)


# In[46]:


# indexing and slicing
## indexing

# single indexing

F = np.array([1, 1, 2, 3, 5, 8, 13, 21])
# print the first element of F
print(F[0])
# print the last element of F
print(F[-1])

# multidimensionl array

A = np.array([ [3.4, 8.7, 9.9], 
               [1.1, -7.8, -0.7],
               [4.1, 12.3, 4.8]])
print(A[1][0])

# numpy way to access multidimensional array

print(A[1, 0])

## slicing

# 1-d
S = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(S[2:5])
print(S[:4])
print(S[6:])
print(S[:])

# m-d

A = np.array([
[11, 12, 13, 14, 15],
[21, 22, 23, 24, 25],
[31, 32, 33, 34, 35],
[41, 42, 43, 44, 45],
[51, 52, 53, 54, 55]])
print(A[:3,2:])
print(A[3:,:])
print(A[:,-1:])
X = np.arange(28).reshape(4, 7)
print(X[::,::3])

A = np.array(
    [ [ [45, 12, 4], [45, 13, 5], [46, 12, 6] ], 
      [ [46, 14, 4], [45, 14, 5], [46, 11, 5] ], 
      [ [47, 13, 2], [48, 15, 5], [52, 15, 1] ] ])
A[1:3, 0:2]  # equivalent to A[1:3, 0:2, :]


# In[48]:


# checkingif array share same memory block using np.may_share_memory(A,B)

A = np.arange(12)
B = A.reshape(3, 4)
A[0] = 42
print(B)
np.may_share_memory(A, B)  # false positive


# ## dtype object and dataTypes

# In[74]:


# creating dtype object using np.dytpe(numpy_type/[(numpy_type),]) --> dtype object

i16 = np.dtype(np.int16)
print(i16)
arr = np.arange(10,dtype = i16)
print(arr)


# In[76]:


# creating column with (column_string,numpy_type)

dt = np.dtype([("density",np.int32)])
arr = np.array([(123,),(234,),(345,)],dtype=dt)
# print(arr)
print(repr(arr))


# In[77]:


# accessing whole column
print(arr['density'])


# In[101]:


# using string type eg. 'i4' to specifiy dtype
# i4 --> 4 byte integer
# '<i4' --> < --> little-endian > --> big-endian
# no sign means machine dependent endiness
dt2 = np.dtype([("country","S20"),("density",'i8'),("area","i4"),("population","i4")])  # binary ASCII string (prefixed with b')
dt2unicode = np.dtype([("country",np.unicode,20),("density",'i8'),("area","i4"),("population","i4")])  # unicode string

arr = np.array([ ('Netherlands', 393, 41526, 16928800),
    ('Belgium', 337, 30510, 11007020),
    ('United Kingdom', 256, 243610, 62262000),
    ('Germany', 233, 357021, 81799600),
    ('Liechtenstein', 205, 160, 32842),
    ('Italy', 192, 301230, 59715625),
    ('Switzerland', 177, 41290, 7301994),
    ('Luxembourg', 173, 2586, 512000),
    ('France', 111, 547030, 63601002),
    ('Austria', 97, 83858, 8169929),
    ('Greece', 81, 131940, 11606813),
    ('Ireland', 65, 70280, 4581269),
    ('Sweden', 20, 449964, 9515744),
    ('Finland', 16, 338424, 5410233),
    ('Norway', 13, 385252, 5033675)],dtype=dt2unicode)
print(arr[:4])  # accessing some row
print(arr["country"])  # accessing column
print(arr[5])  # accessing row
print(arr["country"][5])  # accessing element of column
print(arr["country"][:5])  # accessing element of column
print(arr[6]["country"])  # accessing element of column

d1 = np.dtype('d')
d2 = np.dtype('<d')
d3 = np.dtype('>d')
print(d1.name,d1.byteorder,d1.itemsize)  # using dt.[name,byteorder,itemsize]
print(d2.name,d2.byteorder,d2.itemsize)
print(d3.name,d3.byteorder,d3.itemsize)


# In[118]:


# storing and retriving data from file  np.savetxt(file_name,ndarray,fmt,delimiter)
#                                       np.genfromtxt(file,dtype,delimiter) --> ndarray
#                                       np.loadtxt(file,dtype,converters(dict),delimiter)  --> ndarray

# storing CSV

np.savetxt("file.csv",arr,"%s,%d,%d,%d",delimiter=";")  # ??? use of delimiter

# reading CSV
# using getfromtxt

x = np.genfromtxt("file.csv",dtype=dt2unicode,delimiter=",")
print(x)

# using loadtxt

x2 = np.loadtxt("file.csv",dtype=dt2unicode,converters={0:lambda x:x.decode('utf-8')},delimiter=",")
x3 = np.loadtxt("file.csv",dtype=dt2unicode,converters={0:lambda x:x.decode('utf-8'),1:lambda x:int(x)*2},delimiter=",")  # argument of x is string

print(x3)


# # <u>Numerical Operations</u>

# ### Scalars

# In[122]:


lst = [1,2,3,4,5,6]
narray = np.array(lst,dtype="d")
print(narray)
narray = narray ** 5.5  # any (+-*/**) scalar operation
print(narray)


# In[123]:


# equivalent list approach
lst = [1,2,3,4,5,6]
lst2 = [x+5.5 for x in lst]
print(lst)
print(lst2)


# In[125]:


# performance (IGNORE)

get_ipython().run_line_magic('timeit', 'narray +5.5')
get_ipython().run_line_magic('timeit', '[x+5.5 for x in lst]')


# ### Two arrays

# In[131]:


# operation performed on respective element (not matrix multiplication)
arr1 = np.arange(9).reshape(3,3)
print (arr1)
arr2 = np.identity(3)
print(arr2)
print(arr1+arr2)
print(arr1*(arr2+1)) # (not matrix multiplication)


# ### dot product

# In[132]:


# using np.dot(array,array,out=None)  --> ndarray

print(np.dot(3, 4))  # scalar
x = np.array([3])
y = np.array([4])
print(x.ndim)
print(np.dot(x, y))  # zero dimensional array
x = np.array([3, -2])
y = np.array([-4, 1])
print(np.dot(x, y))  # one-d array


# In[133]:


A = np.array([ [1, 2, 3], 
               [3, 2, 1] ])
B = np.array([ [2, 3, 4, -2], 
               [1, -1, 2, 3],
               [1, 2, 3, 0] ])
# es muss gelten:
print(A.shape[-1] == B.shape[-2], A.shape[1])  # checking condition
print(np.dot(A, B))  # 2-d array


# In[134]:


X = np.array( [[[3, 1, 2],
                [4, 2, 2],
                [2, 4, 1]],
               [[3, 2, 2],
                [4, 4, 3],
                [4, 1, 1]],
               [[2, 2, 1],
                [3, 1, 3],
                [3, 2, 3]]])
Y = np.array( [[[2, 3, 1],
                [2, 2, 4],
                [3, 4, 4]],
            
               [[1, 4, 1],
                [4, 1, 2],
                [4, 1, 2]],
            
               [[1, 2, 3],
                [4, 1, 1],
                [3, 1, 4]]])
R = np.dot(X, Y)  # 3-d array
print("The shapes:")
print(X.shape)
print(Y.shape)
print(R.shape)
print("\nThe Result R:")
print(R)


# ## Numpy Matrix

# In[136]:


# using np.mat(array)  --> matrix

A = np.array([ [1, 2, 3], [2, 2, 2], [3, 3, 3] ])
B = np.array([ [3, 2, 1], [1, 2, 3], [-1, -2, -3] ])
print(A*B,type(A*B))

MA = np.mat(A)
MB = np.mat(B)
print(MA*MB,type(MA*MB))


# ### Comparision/logical operator on Array

# In[137]:


# this compare array elementwise
A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([ [11, 102, 13], [201, 22, 203], [31, 32, 303] ])
C = A==B
print(C,C.dtype)


# In[138]:


# comparing whole array use np.array_equal(arr,arr)  --> bool

print(np.array_equal(A,B))


# In[139]:


# elementwise logical and/or using np.logical_[or/and](arr,arr) -->bool_array

a = np.array([ [True, True], [False, False]])
b = np.array([ [True, False], [True, False]])
print(np.logical_or(a,b))
print(np.logical_and(a,b))


# ### Broadcasting

# #### example1
# <img src="files/broadcasting_example_1.png">

# In[140]:


A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([1, 2, 3])
print("Multiplication with broadcasting: ")
print(A * B)
print("... and now addition with broadcasting: ")
print(A + B)


# #### example2
# <img src="files/broadcasting_example_2.png">

# In[142]:


A = np.array([ [11, 12, 13], [21, 22, 23], [31, 32, 33] ])
B = np.array([1, 2, 3])
B = B[:,np.newaxis]  # using newaxis to interchange row  <-->  column
print(A+B)
print(A*B)


# #### example3
# <img src="files/broadcasting_example_3.png">

# In[144]:


A = np.array([10, 20, 30])
B = np.array([1, 2, 3])
# 1st way

print(A*B[:,np.newaxis])

# 2nd way

print(A[:,np.newaxis]*B)


# # <u>Changing dimension of array</u>
# <img src = "files/axis.png">

# ## <u>Flatten</u> : convert to 1-d array

# In[154]:


# using arr.flatten(order="A|F|C") --> 1-d array

A = np.array([[[ 0,  1],
               [ 2,  3],
               [ 4,  5],
               [ 6,  7]],
              [[ 8,  9],
               [10, 11],
               [12, 13],
               [14, 15]],
              [[16, 17],
               [18, 19],
               [20, 21],
               [22, 23]]])

print(A.flatten(order = "C"))  # C-style order row-first
print(A.flatten(order = "F"))  # fortran style order column-first
print(A.flatten(order = "A"))


# In[152]:


# using arr.ravel(order = "A|C|F|K")  --> 1d array
# using np.ravel(a,order)

print(A.ravel())
print(A.ravel(order="A"))
print(A.ravel(order="F"))
print(A.ravel(order="A"))
print(A.ravel(order="K"))
print(np.ravel(A,order="F"))


# ### Reshape

# In[160]:


#using arr.reshape(newshape_tuple,order="C|F|A")
#using np.reshape(arr,newshape_tuple,order)

X = np.arange(100,dtype = int)
print(X)
Y = np.reshape(X,(2,5,10),order = "C")
print(Y)
Z = X.reshape((5,2,10),order = "F")
print("\n\n",Z)


# ## <u>Concatenating arrays</u>

# In[161]:


# using np.concatenate(arr_tuple,axis=0) --> ndarray

x = np.arange(5);
y = np.arange(5,10);
z = np.arange(10,15);
c = np.concatenate((x,y,z))  #
print(c)


# In[165]:


# concatenating multidimenstional array
# NOTE: dimension should be same
# axis = 0 by default

x = np.arange(50)
y = np.arange(50,100)
x = x.reshape((1,5,10))
y = y.reshape((1,5,10))
print(x,"\n\n",y)
c = np.concatenate((x,y))
print("\n\n",c)


# In[167]:


# axis = 1
print("axis=1\n",np.concatenate((x,y),axis = 1))
# axis = 2
print("\n\naxis=2\n",np.concatenate((x,y),axis = 2))


# ### Adding new dimension

# In[174]:


# using slicing and np.newaxis

# 1d --> 2d
D1 = np.arange(10)
print("1D\n",D1)
D2 = D1[:,np.newaxis]
print("\n2D\n",D2)
D22 = D1[np.newaxis,:]
print("\nalso\n",D22)

# 2d --> 3d

D2 = np.arange(50).reshape((5,10))
print("\n2D\n",D2)
print("\n3D\n",D2[np.newaxis,:,:])  # new axis at 0-axis
print("\n3D another\n",D2[:,np.newaxis,:])  # new axis at 1-axis
print("\n3D also\n",D2[:,:,np.newaxis])  # new axis at 2-axis


# ### Vector stacking

# In[183]:


# using np.row_stack(arr_tuple)
#       np.column_stack(arr_tuple)
#       np.dstack(arr_tuple)
# dimension should be same

# 1-d array
A = np.array([1,2,3])
B = np.array([4,5,6])
print(np.row_stack((A,B)),"\n\n")
print(np.column_stack((A,B)),"\n\n")
print(np.dstack((A,B)),"\n\n")

# 2-d array

A = np.array([[1,2,3],[4,5,6]])
B = np.array([[7,8,9],[10,11,12]])
print("\n\n",A,"\n\n",B,"\n\n")
print(np.row_stack((A,B)),"\n\n")
print(np.column_stack((A,B)),"\n\n")
print(np.dstack((A,B)),"\n\n")


# ### Repeating pattern using "tile()"

# In[184]:


# using np.tile(arr,reps)  --> reps tuple

x = np.array([ [1, 2], [3, 4]])
print(np.tile(x,(5,4)))


# # <u>Probability</u>

# ## <u>Random</u> number and choices

# ### Random Numbers

# In[187]:


# using random.random() --> [0,1)
# not cyptographically safe
import random
rnd = random.random()
print(rnd)


# In[191]:


# using SystemRandom class --> [0,1)
# cryptographically safe
cls = random.SystemRandom()
print(cls.random())


# In[201]:


# PRACTICE: generate list of random numbers
def random_list(n,secure = 0):
    if secure == 0:
        rnd = random
    elif secure == 1:
        rnd = random.SystemRandom()
    else:
        rnd = np.random
    return [rnd.random() for x in range(n)]
# print(random_list(20))
# print(random_list(20,True))
# print(random_list(40))
# print(random_list(40,True))

# performance testing
get_ipython().run_line_magic('timeit', 'random_list(100)')
get_ipython().run_line_magic('timeit', 'random_list(100,1)')
get_ipython().run_line_magic('timeit', 'random_list(100,2)')
get_ipython().run_line_magic('timeit', 'np.random.random(100)')
get_ipython().run_line_magic('timeit', 'random.random()')
sec = random.SystemRandom()
get_ipython().run_line_magic('timeit', 'sec.random()')
get_ipython().run_line_magic('timeit', 'np.random.random()')


# In[203]:


# using numpy random number generator np.random.random(list_len)  --> ndarray
# not secure
print(np.random.random())
print(np.random.random(10))


# #### Sum to one
# <i>p1+p2+p3+...+pn = 1</i>

# In[205]:


# using arr.sum() to add array
# using arr/sum to normalize to 1

rnarr = np.random.random(50)
print(rnarr)
rnsum = rnarr.sum()
print(rnsum)
prob = rnarr/rnsum
print(prob,"\n\n",prob.sum())


# In[224]:


# PRACTICE: create random password creator
def passwd(n,t=3):

    lst = ["abcdefghijklmnopqrstuvwxyz","ABCDEFGHIJKLMNOPQRSTUVWXYZ","0123456789","_@#$&*!%"]
    l2 = [len(lst[0]),len(lst[1]),len(lst[2]),len(lst[3])]
    sr = random.SystemRandom()
    l = []
    while(n>0):
        r = sr.randint(0,t)
        l.append(lst[r][sr.randint(0,l2[r]-1)])
        n-=1
    return "".join(l)
print(passwd(100,1))


# #### Random Integer

# In[226]:


# using "PURE" python
# using random.randint(start,end)
# NOTE: it gives [start,end)

print([random.randint(1,6) for _ in range(6)])


# In[230]:


# using numpy np.random.randint(start,end,arr_size)

print(np.random.randint(1,7))  # without size, size = None
print(np.random.randint(1,7,size = 1))  # size = 1
print(np.random.randint(1,7,size = 6))  # size = 6
print(np.random.randint(1,7,size = (3,6)))  # 3x6 array


# 
# ### Choices and samples

# In[235]:


# using random.choice(seq)  --> random_element
# pythonic way

lst = ["India","Sri Lanka","Nepal","China","Bhutan","Bangladesh"]
tupl = ("India","Sri Lanka","Nepal","China","Bhutan","Bangladesh")
strng = "roshansingh"

print(random.choice(lst))
print(random.choice(tupl))
print(random.choice(strng))


# In[253]:


# using np.random.choice(seq,size_tuple,replace=true)
# seq cant be string
# when replace = false then produce of dimension (size_tuple) must be <= sizeof seq
# ex size_tuple = (2,3): product = 2x3=6:sizeof seq = 6: since seq == produce: no error: else error

print(np.random.choice(lst))  # just like python's
print(np.random.choice(lst,size=5))  # 1-d array shape=(5,)
print(np.random.choice(tupl,size=(2,3)))  # 2-d array shape=(2,3)
print(np.random.choice(lst,size=(2,3),replace = False))


# In[254]:


#SAMPLE


# In[273]:


# using pythonic sample random.sample(population,k) --> list
#populatin can be sequence or set

lst = ["India","Sri Lanka","Nepal","China","Bhutan","Bangladesh"]
tupl = ("India","Sri Lanka","Nepal","China","Bhutan","Bangladesh")
strng = "roshansingh"
st = set(lst)

print(random.sample(lst,5))
print(random.sample(tupl,4))
print(random.sample(strng,3))
print(random.sample(st,2))


# In[268]:


# using numpy sample np.random.random_sample(size_tuple)
# returns [0,1)

print(np.random.random_sample(5))  # size = 5: 1d array of 5 element
print(np.random.random_sample((5,)))  # size = (5,): 1d array of 5 element
print(np.random.random_sample((2,5)))  # size = 5: 2d array of 5 element
print("\n\n")
# scalar operations
a = 5
b = 10
print((b-a)*np.random.random_sample(10)+a)  # prints float b/w [5,10)


# <h2><u>Boolean Indexing</u></h2>

# In[7]:


arr = np.array([1,2,3,4,5,6])  # 1d array
print(arr)
print(arr%2==0)

arr2 = np.array([[0,1,2,3,4,5,6,7,8,9],
                [10,20,30,40,50,60,70,80,90,100]])  # 2d array
print(arr2)
print(arr2%3==0)



# In[9]:


# using ndArray.astype(np.int) to convert bool to int

B = arr%2==0
print(B)
print(B.astype(np.int))


# <h3>Fancy Indexing</h3>

# In[16]:


# using fancy indexing to access element using boolean index and creating copy
# note: No of Element in boolean ndarray should equal to that of ndarray we are going to access
C = np.array([123,188,190,99,77,88,100])
A = np.array([0,1,2,3,4,5,6])
D = C[A%2==0]
print(C)
print(A%2==0)
print(D)


# ### Indexing With Integer array

# In[17]:


C[[0, 2, 3, 1, 4, 1]]


# ### Nonzero and Where

# In[20]:


#nonzero: returns indices of nonzero element of ndarray
#syntax: np.nonzero(ndarray) or ndarray.nonzero()

a = np.array([[0, 2, 3, 0, 1],
              [1, 0, 0, 7, 0],
              [5, 0, 0, 1, 0]])
print(a.nonzero())
print(np.nonzero(a))

# transpose

print(np.transpose(a.nonzero()))
# getting nonzero elements
a[np.nonzero(a)]


# <h1><u>Reading and Writing to file</u></h1>

# In[23]:


# saving ndarray to file
# np.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='', comments='# ')

x = np.array([[1, 2, 3], 
              [4, 5, 6],
              [7, 8, 9]], np.int32)
np.savetxt("test.txt", x)

#reading from saved files
y = np.loadtxt("test.txt", delimiter=" ")
print(y)


# <small><i>for more see doc</i></small>

# In[ ]:




