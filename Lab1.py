
'''
Lab 1 Solution
Thanks to Kyle Crockett
'''

########## Part 1 ###########

'''
   1) Create a variable, x, and set its value to 16
      Create a variable, y, and set its value to square root of x
      Divide x by three fifths of y, and store the result in x (hint: /=)

'''
# YOUR CODE GOES HERE
import math

x = 16
y = math.sqrt(x)

x /= ((3*y)/5) # Yes order of operations handles this without parentheses but I use them for clarity. 

'''
    2)  A cube has the edge defined below.

    Store its space diagonal, surface area and volume in new variables.
    Print out the results and check your answers.
    Change the value of the edge and ensure the results are still correct.
'''

edge = 10

# YOUR CODE GOES HERE

import math

diag = 3 * math.sqrt(edge)
area = 6 * (edge**2)
vol = edge**3

print("Space diagonal: " + str(diag))
print("Area: " + str(area))
print("Volume: " + str(vol))


######### Part 2 ###########

'''
    1)  For each of the following Python expressions, write down the output value. If it would give an error then write error.

                (a)False == 0
                
                (b) True != 1
                
                (c) 40 // 2
                
                (d) 41 // 2
                
                (e) 41 % 2
                
                (f) True + 1.33
                
                (g) False - `True'
                
                (h) False + \True"
                
                (i) 15/7+5*8**4
                
                (j) ('Hello' == 'Hi') or ( 12 > -6 )

'''
# YOUR CODE GOES HERE
'''
(a) True

(b) False

(c) 20

(d) 20

(e) 1

(f) 2.33

(g) As I typed a string "TypeError: unsupported operand type(s) for -: 'bool' and 'str'"
(g) When Copied: "SyntaxError: invalid syntax" due to ` not '

(h) SyntaxError: unexpected character after line continuation character

(i) 20482.14285714286

(j) True

'''



######### Part 3 ###########


'''
    1) write a python code to calculate the age based on the user's birthdate (the year of birth, e.g.: age = 1989), if age is greater than 18 then outputs “adults' category” otherwise outputs “Children Category”.

'''

year = 1989
# YOUR CODE GOES HERE

from datetime import date

current_year = date.today().year
age = current_year - year

if (age > 18):
    print("adults' category")
else:
    print("Children Category")



'''
	2) Repeat q1:
    If the year is not within 1910-2020 then prints an error message for the user
    Otherwise: calculates his/her age, if age is greater than 18 then outputs “adults' category” otherwise outputs “Children Category”.

'''

year1 = 1989
year2 = 1857

year = year2
# YOUR CODE GOES HERE

from datetime import date
import sys

if (year < 1910 or year > 2020):
    sys.exit("Error: Birth year must fall within 1910-2020")

current_year = date.today().year
age = current_year - year

if (age > 18):
    print("adults' category")
else:
    print("Children Category")
    
######### Part 4 ###########



'''
    1) Write a python code to print all the perfect square numbers less than 300.
'''
# YOUR CODE GOES HERE
i = 1

while i**2 < 300:
    print(i**2)
    i += 1


'''
    2) Write a python code to print all the perfect square numbers less than 300 and greater than 20 except for 100 and 121.
'''
# YOUR CODE GOES HERE

i = 1

while i**2 < 300:
    if i**2 > 20 and i**2 < 300 and i**2 != 100 and i**2 != 121:
        print(i**2)
    i += 1


'''
    3) Write a python code to calculate 100*101*102...*200
'''
# YOUR CODE GOES HERE

i = 100
num = 1

while i < 201:
    print(i)
    num *= i
    i += 1

print(num)

######### Part 5 ###########

'''
    1) Given a list of values: x = [1,'ok',3, 17.01, True]
    Write a code to print the last element of it
'''
# YOUR CODE GOES HERE

x = [1,'ok',3, 17.01, True]
print(x[-1])

'''   2) Given a list of integers: e.g.: [1,2,3,2,0]
        (a) return the average
        (b) return the list resulted from adding up each number with its index. e.g.: output:[1,3,5,5,4]
        (c) given another list, return their common elements. e.g.: SecondList = [1; 1; 2; 2; 2; 2; 4; 6; 7; 88; 8], output :[1,2]
      
        
'''
  # YOUR CODE GOES HERE

x = [1,2,3,2,0]
print("Average: " + str(sum(x)/len(x)))


y = []
for i, val in enumerate(x):
    y.append(i+val)
print("'x' plus index: " + str(y))


SecondList = [1, 1, 2, 2, 2, 2, 4, 6, 7, 88, 8]
x_set = set(x) 
SecondList_set = set(SecondList)
print("Common elements: " + str(x_set.intersection(SecondList_set)))


    
######### Part 6 ###########

'''
    1)  Write a function to find the even numbers in the list and return a list of those numbers:
        e.g.: list1 = [9,-6, 0, 7, 1, 5, 6, 8]-->[-6, 0, 6, 8]
'''

# YOUR CODE GOES HERE

def rtn_even(l):
    ev = []
    for n in l: 
        if n % 2 == 0:
            ev.append(n)
    return ev


'''
    2)  Write a function to find the odd numbers in the list and return a list of their indices: e.g.: list1 = [9,-6, 0, 7, 1, 5, 6, 8] --> [0,3,4,5]
'''
# YOUR CODE GOES HERE

def rtn_odd_idx(l):
    odd_idx = []
    for i, n in enumerate(l): 
        if n % 2 != 0:
            odd_idx.append(i)
    return odd_idx


######### Part 7 ###########
'''
    1) Write a function to get a message as an input and to replace all instances of ‘o’ with ’a’ and to return the updated message.
'''

# YOUR CODE GOES HERE

def rep_o(s):
    ns = ""
    for i, character in enumerate(s):
        if character == 'o':
            ns += 'a'
        else:
            ns += character
    return ns
        

######### Part 8 ###########
'''
    1) Write a function to drop the duplications in a list of numbers.
    (Hint: use dictionary)  e.g.: 
    Input_list = [11,2,3,8,0,11,4,2,2,7,0]-->[11,2,3,8,0,4,7]

'''
# YOUR CODE GOES HERE

def drop_dup(l):
  return list(dict.fromkeys(l))


######### Part 9 ###########
'''
    1) Write a function to get the radius of a circle and to return its area. (import pi and exponentiation from math module)
    
'''    
# YOUR CODE GOES HERE

import math

def circ_area(r):
    return math.pi*(r**2)


######### Part 10 ###########  
'''
   1) Write a python function to find the frequency of the characters in a sentence.
(Hint : Use a dictionary)

e.g.:  ‘Hhellloo’   -->  {‘H’:1 , ‘h’: 1, ‘e’:1, ‘l’:3 , ‘o’:2}
    
'''  
# YOUR CODE GOES HERE

def ch_freq(s):
    d = {} 
    for i in s: 
        if i in d: 
            d[i] += 1
        else: 
            d[i] = 1
    return d