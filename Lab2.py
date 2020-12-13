
'''
Lab 2 solution
'''


##########Part 1 ###########


'''
    1) download iris-data.csv (available under Week 5 module on Canvas), load the data using pandas.read_csv and print it

'''
# YOUR CODE GOES HERE
import pandas as pd
with open('iris-data.csv') as csvfile:
    data = pd.read_csv(csvfile, delimiter = ',')

'''
    2) after loading the data, print out index, columns, and values information and their types (in a comment line expalin them)
       print out the length of the data set 
       print out the last 50 data points
       print out the labels 
       print out the labels and petal_length
'''
# YOUR CODE GOES HERE

print(type(data),data.index,data.columns,data.values,len(data))
print(data[-50:])
print(data['species'])
print(data[['sepal_length', 'sepal_width']])

'''
    3) print out the mean and std of each feature
    print out the mean of petal_length for first 100 samples
    print out the maximum and minimum values for each feature

'''
# YOUR CODE GOES HERE
print(data.mean())
print(data.std())
print(data.loc[0:100, "petal_length"].mean())
print(data.max())
print(data.min())
'''
    4)  print out the frequency count of each label
    Hint: use pandas’ function value_counts 

'''
# YOUR CODE GOES HERE
print(data["species"].value_counts())
######### Part 2 ###########

'''
    1) use pandas.DataFrame.drop_duplicates to drop duplications in "petal_length" feature (keep the last instance) and print out the resulted data
    print out the length of the data set 
    print out the mean of each feature
    print out the frequency/count of each label

'''
# YOUR CODE GOES HERE
data_new = data.drop_duplicates(['petal_length'], keep='last')
print(len(data_new))
print(data_new.mean())
print(pd.value_counts(data_new["species"]))
'''
    2)  plot the original data in a single graph
    Hint: pandas.DataFrame.plot

'''
# YOUR CODE GOES HERE
data.plot()

'''    3)  plot the bar graph of the data    
        Hint: pandas.DataFrame.plot.bar()
'''

# YOUR CODE GOES HERE
data.plot.bar()

'''    4)  plot the histogram graph for "petal_length" feature    

        Hint: pandas.DataFrame.plot.histograms()
'''
# YOUR CODE GOES HERE
data['petal_length'].plot.hist()

'''    5)  plot the bar graph to show the frequency of each label 

'''
# YOUR CODE GOES HERE
pd.value_counts(data["species"]).plot.bar()