import numpy as np
import matplotlib.pyplot as plt
 
  
# function to add value labels
def addlabels(x,y):
    for i in range(len(x)):
        #plt.text(i, y[i], y[i], ha = 'center')
        plt.text(i, y[i]//2, y[i], ha = 'center')
        
# creating the dataset
data = {'A01':73,'A02':69,'A03':70,'A04':78,'A05':82,'A06':92,'A07':85,'A08':80,'A09':63}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='pink',
        width = 0.4)

addlabels(courses, values)
 
plt.xlabel("Patients")
plt.ylabel("Time taken (in min)")
plt.title("Time taken to calcuate results of each patient")
plt.show()