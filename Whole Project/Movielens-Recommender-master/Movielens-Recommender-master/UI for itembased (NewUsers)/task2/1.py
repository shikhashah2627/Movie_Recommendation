#!/usr/bin/python

import csv

f = open('/home/prerna/Desktop/Spring 2018/CS445/Project/UI for userbased(exist user)/movie_recommender/Items.csv')
csv_f = csv.reader(f)

newf = []

for row in csv_f:
	nr = row
	#print(len(row))
	flag = False
	index  = -1
	#print nr
	for i in range(1,len(row)):
		
		if flag:
			nr[i] = '0'

		if (nr[i] == '1'):
			flag = True
	#print(nr)
	newf.append(nr)
print(newf)

with open('/home/prerna/Desktop/Spring 2018/CS445/Project/UI for userbased(exist user)/movie_recommender/Items.csv', 'w') as f:
    # Overwrite the old file with the modified rows
    writer = csv.writer(f)
    writer.writerows(newf)
    print("done")
