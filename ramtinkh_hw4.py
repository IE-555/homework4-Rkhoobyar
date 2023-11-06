# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:35:32 2023

@author: ramtin
"""

import pandas as pd  # Import pandas for handling dataframes
import numpy as np  # Import np for getting the mean of some lists later on
import seaborn as sns  # Imported the library for the part 2 of the homework
import matplotlib . pyplot as plt #required library for plotting is imported
df=pd.read_csv('scatter_data.csv',header=None, comment='%')  # The data is read here from the sample for problem 1
x=df.iloc[:,0]  # The first column is the x
y=df.iloc[:,1]  # The second column is the y
plt.scatter(x,y,color="green",marker="^")  # Scatter plot of the x and y with green triangles
minx=x.min()  # min and max is identified here
maxx=x.max()
row_min=x.idxmin()  # The index for min and max is identified here
row_max=x.idxmax()
list_x=[minx, maxx]  # The list for x coordinates and y coordinates for the line is created here
list_y=[y.iloc[row_min],y.iloc[row_max]]
plt.plot(list_x, list_y,color="red", linestyle="dashed")  # The dashed red line connecting the min x and max x is plotted here
plt.title ('Widget Measurements')  # Title, labels and legends are added here
plt.xlabel("x [inches]")
plt.ylabel("y [inches]")
plt.legend(["Observation","extreme x points"])
plt.savefig ('Widget Measurements.png')  # The plot is saved and closed for the next plot
plt.close()




ef=pd.read_csv('student_grades.csv',header=None, comment='%')  # The sample for the 2nd problem is read here
grade=[]  # This is an empty list later on 1(F), 2(D), 3(C), 4(B) and 5(A) for the data based on the ranges presented in the table in the question statement is added here
score=ef.iloc[:,1]  # The 2nd column of the dataframe is stored here
for i in score:  # The data are assigned to their grade based on their value
    if i < 60 :
        grade.append(1)
    elif i >= 60 and i < 70 :
        grade.append(2)
    elif i >= 70 and i < 80 :
        grade.append(3)
    elif i >= 80 and i < 90 :
        grade.append(4)
    elif i >= 90 and i <= 100 :
        grade.append(5)       
counts, edges, bars=plt.hist(grade, bins=[1,2,3,4,5,6], align='left', rwidth=0.8, edgecolor='black', color='orange')  # Histogram with orange bars and black edges are created here
plt.bar_label(bars)  # The frequency of each bin is added on top of it 
plt.xticks(range(1, 6), ['F', 'D', 'C', 'B', 'A'])  # The ticks are changed to A, B, C, D and F
plt.title ('Grade Distribition')  # The title and labels are added here 
plt.xlabel("Grade")
plt.ylabel("Count")
plt.savefig ('Grade Distribition.png')  # Figures are saved and closed preparing for the next plot
plt.close()




gf=pd.read_csv('solution_data.csv',header=None, comment='%')  # The data is read for the sample excel file
genetic=[]  # The data with the genetic algorithm are added here
simulated=[]  # The data with the simulated annealing are added here
tabu=[]  # The data with tabu search is added here
optimal=[]  # The optimal data are stored here
for i in gf.index:  # The above lists are filled with their corresponding data
    if gf.iloc[i,1] == 'genetic algorithm':
        genetic.append(gf.iloc[i,2])
for i in gf.index:
    if gf.iloc[i,1] == 'simulated annealing':
        simulated.append(gf.iloc[i,2])
for i in gf.index:
    if gf.iloc[i,1] == 'tabu search':
        tabu.append(gf.iloc[i,2])        
for i in gf.index:
    if gf.iloc[i,1] == 'optimal':
        optimal.append(gf.iloc[i,2])
gap1=[]  # The gap value requested in the question statement is added here for the genetic algorithm
gap2=[]  # The one with the simulated annealing is added here
gap3=[]  # The one with tabu search is added here
for j, g in enumerate(genetic):  # The value for each specific alorithm is calculated for each row and added to the above lists 
    gap1.append(abs(((optimal[j]-g)/optimal[j])*100))
for j, g in enumerate(simulated):
    gap2.append(abs(((optimal[j]-g)/optimal[j])*100))
for j, g in enumerate(tabu):
    gap3.append(abs(((optimal[j]-g)/optimal[j])*100)) 

y= [np.mean(gap1),np.mean(gap2),np.mean(gap3)]  # The y of the bar plot which is the average of the gap values is calcualted and listed here
x= ['genetic algorithm', 'simulated annealing', 'tabu search']  # The names of the algorithms are stored for x
fig , ( ax1 , ax2 ) = plt . subplots (1,2, sharey= True , figsize=(15,4) )  # The 2 horizontal subplots are created here
ax1.bar(x, y, width =0.8, edgecolor='black', color='orange' )  # The barplot is created here for the left subplot
ax2.boxplot([gap1,gap2,gap3])  # The box plot is created here 
plt.xticks([1, 2, 3], ['genetic algorithm', 'simulated annealing', 'tabu search'])
fig.suptitle ('Comparison of Optimality Gaps for Heuristics')
ax1.set( xlabel ='Heuristic Method', ylabel ='Optimality Gap(%)')
ax2.set( xlabel ='Heuristic Method')
plt.savefig ('Comparison of Optimality Gaps for Heuristics.png')  # Plot is saved and closed 
plt.close()


  
hf=pd.read_csv('IRIS.csv') # Data was found from https://www.kaggle.com/datasets/arshid/iris-flower-dataset
# draw lineplot a function in the Seaborn library, which is built on top of
# Matplotlib and provides an easy interface to create various statistical 
#visualizations, including line plots
sns.lineplot(x="sepal_length", y="sepal_width", data=hf)  # Tutorial can be found here :
# https://seaborn.pydata.org/generated/seaborn.lineplot.html
  
# setting the title and labels here
plt.title('Lenght and Width of IRIS Flower')
plt.xlabel("Sepal Lenght")
plt.ylabel("Sepal Width")
plt.savefig ('Lenght and Width of IRIS Flower.png')  # Plot is saved and closed
plt.close()