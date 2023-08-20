import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib import rcParams
import os

def log_list(a):
    for i in range(len(a)):
        a[i] =  math.log(a[i],10)
    return a

labels = ['16', '64', '256', '512', '2048', '4096']


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
file_path = current_directory+"/eucliddist.out"
eva = []
heco = []
heir = []

try:
    with open(file_path, "r") as file:
        for line in file:
            if "EVA" in line:
                start_index = line.find("Time: ") + len("Time: ")
                end_index = line.find("ms", start_index)
                if start_index != -1 and end_index != -1:
                    time_str = line[start_index:end_index].strip()
                    eva.append(float(time_str) / 1000)
            elif "HECO" in line:
                start_index = line.find("Time: ") + len("Time: ")
                end_index = line.find("ms", start_index)
                if start_index != -1 and end_index != -1:
                    time_str = line[start_index:end_index].strip()
                    heco.append(float(time_str) / 1000)
            elif "HEIR" in line:
                start_index = line.find("Time ") + len("Time ")
                end_index = line.find("ms", start_index)
                if start_index != -1 and end_index != -1:
                    time_str = line[start_index:end_index].strip()
                    heir.append(float(time_str) / 1000)

except FileNotFoundError:
    print("Result File does not exist")


width = 0.25  
x1_list = []
x2_list = []
x3_list = []
label_list = []
for i in range(len(eva)):
    eva[i] = math.log10(eva[i])+3
    heco[i] = math.log10(heco[i])+3
    heir[i] = math.log10(heir[i])+3
    x1_list.append(i)
    x2_list.append(i + width)
    label_list.append(i + width + width/2)
    x3_list.append(i + width + width)


plt.ylabel('Latency (s)',fontdict={'size'   : 20})  
plt.xlabel('Length of the vector',fontdict={'size'   : 20})  

plt.bar(x1_list, eva, width=width, color='#EEDEB0', align='edge',hatch='-', label='EVA')
plt.bar(x2_list, heco, width=width, color='#9B4400', align='edge',hatch='x', label='HECO')
plt.bar(x3_list, heir, width=width, color='#1E90FF', align='edge', label='HEIR')
plt.ylim(0, 5.1)
plt.yticks([0, 1, 2, 3, 4],labels = ['$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '$1$','$10^1$'], math_fontfamily='cm', size = 16)
plt.xticks(label_list, labels, size = 16)


plt.legend(prop =  {'weight':'normal','size':12},loc='upper left')
plt.tight_layout()
plt.savefig(current_directory+'/eucliddist.pdf')
# plt.show()
