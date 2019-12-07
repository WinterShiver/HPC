import math
map_log = lambda lst: list(map(math.log, lst))

log_scale = [7, 8, 9, 10, 11]
scale = ['$2^{' + str(elem) + '}$' for elem in log_scale]
x_label = []
for elem in scale:
	x_label.append('')
	x_label.append(elem)



y1 = [0.00605488, 0.0294092, 0.208391, 1.74074, 13.8348]
y2 = [0.00615766, 0.019629, 0.119481, 1.02355, 7.53213]
y4 = [0.00475181, 0.0268388, 0.126077, 1.11406, 7.80507]
y8 = [0.00942505, 0.034541, 0.156109, 1.15209, 7.98609]


import matplotlib.pyplot as plt

plt.title('Running Time Comparison\n(Using 2 Cores)')
plt.xlabel('Data Scale')
plt.ylabel('Time (s)')
 
plt.plot(log_scale, y1,'b', label='1 Thread')
plt.plot(log_scale, y2,'g', label='2 Threads')
plt.plot(log_scale, y4,'r', label='4 Threads')
plt.plot(log_scale, y8,'purple', label='8 Threads')
# plt.plot(log_scale, map_log(y16),'brown', label='16 Threads')

loc, labels = plt.xticks()
plt.xticks(loc, x_label, rotation=0)
# loc, labels = plt.yticks()
# plt.yticks(loc, ['$10^{' + str(int(elem)) + '}$' for elem in list(loc)], rotation=0)

'''plt.title('Running Time Comparison when Applying Master-Slave Mode \n (Using 4 Cores)')
plt.xlabel('Data Scale')
plt.ylabel('Time (s)')
 

plt.plot(log_scale, y314,'g', label='Collaborative Computing')
plt.plot(log_scale, y34,'b', label='Master-Slave')

plt.xticks(log_scale, scale, rotation=0)'''

'''plt.title('Running Memory Comparison\n(R stands for Row Seperation, B stands for Block Seperation)')
# plt.xlabel('')
plt.ylabel('Memory (MB)')
 
plt.bar([i for i in range(len(mem2048))], mem2048, width = 0.25,facecolor = 'r', edgecolor = 'white', label = 'Total')
plt.bar([i + 0.3 for i in range(len(mem2048))], mem2048_main, width = 0.25,facecolor = 'g', edgecolor = 'white', label = 'Master')
plt.bar([i + 0.6 for i in range(len(mem2048))], mem2048_sub, width = 0.25,facecolor = 'b', edgecolor = 'white', label = 'Slave')
x_label = [' ', 'R-Send-Recv', ' ', 'R-Scatter-Gather', ' ', 'B-Send-Recv\nRows & Columns', ' ', 'B-Send-Recv\nCannon']
loc, labels = plt.xticks()
plt.xticks([elem + 0.3 for elem in list(loc)], x_label, rotation=0)'''

'''plt.title('Running Time\n(In Calculating 1024 * 1024 Matrixes, Using 4 Cores)')
plt.xlabel('Number of Processes')
plt.ylabel('Time (s)')

time = [34.4074, 18.9103, 11.4054, 12.4829, 12.9519]
corenum = [1, 2, 4, 8, 16]
 
plt.plot(corenum, time,'r')'''

plt.legend(bbox_to_anchor=[0.5, 0.9])
plt.grid()
plt.show()