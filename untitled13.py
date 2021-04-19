
import pandas as pd

d = {
    'value':[20.45,22.89,32.12,111.22,33.22,100.00,99.99],
    'product':['table','chair','chair','mobile phone','table','mobile phone','table'],
     'city':['jp','ko','jp','al','ko','jp','al']
}

df = pd.DataFrame(d)
g1 = df.groupby("product").groups

print(g1)


g2 = df.groupby("city")["value"].max()


g2 = df.groupby("city")["value"].sum()
g2 = df.groupby("city")["value"].min()
g2 = df.groupby("city")["value"].mean()
g2 = df.groupby("city")["value"].count()
g2 = df.groupby("city")["value"].describe()



df = pd.DataFrame(d)
g3 = df.groupby("product").groups
g4 = df.groupby("product")
g5=g4.get_group("table")






import matplotlib.pyplot as plt
import numpy as np

xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints, 'o')
plt.show()

plt.plot(ypoints, 'o:r')
plt.show()


plt.plot(ypoints, marker = 'o', ms = 20)

plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r')

plt.plot(ypoints, marker = 'o', ms = 20, mfc = 'r')

plt.plot(ypoints, marker = 'o', ms = 20, mec = 'r', mfc = 'r')


plt.plot(ypoints, marker = 'o', ms = 20, mec = '#4CAF50', mfc = '#4CAF50')
plt.show()



ypoints = np.array([3, 8, 1, 10])
plt.plot(ypoints, linestyle = 'dashed')
plt.plot(ypoints, color = 'r')
plt.plot(ypoints, c = '#4CAF50')
plt.plot(ypoints, linewidth = '20.5')
plt.plot(ypoints, linestyle = 'dotted')
plt.show()


import numpy as np
import matplotlib.pyplot as plt

x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

font1 = {'family':'serif','color':'blue','size':20}
font2 = {'family':'serif','color':'darkred','size':15}

plt.title("Sports Watch Data", fontdict = font1)
plt.xlabel("Average Pulse", fontdict = font2)
plt.ylabel("Calorie Burnage", fontdict = font2)

plt.plot(x, y)
plt.show()


import matplotlib.pyplot as plt
import numpy as np

ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, np.pi*2, 0.05)

y = np.sin(x)
plt.plot(x,y)
plt.plot(x,y,"r")


plt.plot(x,y,"r")
plt.xlabel("angle")
plt.ylabel("sine")
plt.title('sine wave')
plt.show()





from matplotlib import pyplot as plt
from matplotlib import style
 

x = [5,8,10]
y = [12,16,6]
x2 = [6,9,11]
y2 = [6,15,7]
plt.plot(x,y,'g',label='line one', linewidth=5)
plt.plot(x2,y2,'c',label='line two',linewidth=5)
plt.title('Epic Info')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.legend()
plt.grid(True,color='k')
plt.show()




from matplotlib import pyplot as plt
 
plt.bar([0.25,1.25,2.25,3.25,4.25],[50,40,70,80,20],
label="BMW",width=.5)
plt.bar([.75,1.75,2.75,3.75,4.75],[80,20,20,50,60],
label="Audi", color='r',width=.5)
plt.legend()
plt.xlabel('Days')
plt.ylabel('Distance (kms)')
plt.title('Information')
plt.show()




import matplotlib.pyplot as plt
population_age = [22,55,62,45,21,22,34,42,42,4,2,102,95,85,55,110,120,70,65,55,111,115,80,75,65,54,44,43,42,48]
bins = [0,10,20,30,40,50,60,70,80,90,100]
plt.hist(population_age, bins, histtype='bar', rwidth=0.8)
plt.xlabel('age groups')
plt.ylabel('Number of people')
plt.title('Histogram')
plt.show()



import matplotlib.pyplot as plt
x = [1,1.5,2,2.5,3,3.5,3.6]
y = [7.5,8,8.5,9,9.5,10,10.5]
 
x1=[8,8.5,9,9.5,10,10.5,11]
y1=[3,3.5,3.7,4,4.5,5,5.2]
 
plt.scatter(x,y, label='high income low saving',color='r')
plt.scatter(x1,y1,label='low income high savings',color='b')
plt.xlabel('saving*100')
plt.ylabel('income*1000')
plt.title('Scatter Plot')


plt.legend()
plt.show()




import matplotlib.pyplot as plt
days = [1,2,3,4,5]
  
sleeping =[7,8,6,11,7]
eating = [2,3,4,3,2]
working =[7,8,7,2,2]
playing = [8,5,7,8,13]
  
plt.plot([],[],color='m', label='Sleeping', linewidth=5)
plt.plot([],[],color='c', label='Eating', linewidth=5)
plt.plot([],[],color='r', label='Working', linewidth=5)
plt.plot([],[],color='k', label='Playing', linewidth=5)
  
plt.stackplot(days, sleeping,eating,working,playing, colors=['m','c','r','k'])
  
plt.xlabel('x')
plt.ylabel('y')
plt.title('Stack Plot')
plt.legend()
plt.show()




import matplotlib.pyplot as plt
 
days = [1,2,3,4,5]
 
sleeping =[7,8,6,11,7]
eating = [2,3,4,3,2]
working =[7,8,7,2,2]
playing = [8,5,7,8,13]
slices = [7,2,2,13]
activities = ['sleeping','eating','working','playing']
cols = ['c','m','r','b']
 
plt.pie(slices,labels=activities,colors=cols,startangle=90,shadow= True,explode=(0,0.1,0,0),
  autopct='%1.1f%%')
 
plt.title('Pie Plot')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
 
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)
plt.subplot(221)
plt.plot(t1, f(t1), 'bo', t2, f(t2))
plt.subplot(222)
plt.plot(t2, np.cos(2*np.pi*t2))
plt.show()



