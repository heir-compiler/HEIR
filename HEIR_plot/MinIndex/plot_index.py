import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from matplotlib import rcParams
import os
import sys

def log_list(a):
    for i in range(len(a)):
        a[i] =  math.log(a[i],10)
    return a

args = sys.argv

labels = ['2', '4', '8']

if args[1] == "full":
    labels = ['2', '4', '8', '16', '32']


current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
file_path = current_directory+"/minindex.out"
transpiler = []
heir = []

try:
    with open(file_path, "r") as file:
        for line in file:
            if "Transpiler" in line:
                start_index = line.find("Time ") + len("Time ")
                end_index = line.find("ms", start_index)
                if start_index != -1 and end_index != -1:
                    time_str = line[start_index:end_index].strip()
                    transpiler.append(float(time_str) / 1000)
            elif "HEIR" in line:
                start_index = line.find("Time ") + len("Time ")
                end_index = line.find("ms", start_index)
                if start_index != -1 and end_index != -1:
                    time_str = line[start_index:end_index].strip()
                    heir.append(float(time_str) / 1000)

except FileNotFoundError:
    print("Result File does not exist")


width = 0.3  
x1_list = []
x2_list = []
label_list = []
for i in range(len(heir)):
    transpiler[i] = math.log10(transpiler[i])
    heir[i] = math.log10(heir[i])
    x1_list.append(i)
    x2_list.append(i + width)
    label_list.append(i + width + width/2)


plt.ylabel('Latency (s)',fontdict={'size'   : 20})  
plt.xlabel('Length of the vector',fontdict={'size'   : 20})  

plt.bar(x1_list, transpiler, width=width, color='#87CEFA', align='edge',hatch='x', label='Transpiler')
plt.bar(x2_list, heir, width=width, color='#1E90FF', align='edge', label='HEIR')
plt.yticks([0, 1, 2, 3, 4],labels = ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'], math_fontfamily='cm', size = 16)
plt.xticks(label_list, labels, size = 16)


plt.legend(prop =  {'weight':'normal','size':12},loc='upper left')
plt.tight_layout()
plt.savefig(current_directory+'/minindex.pdf')
# plt.show()
