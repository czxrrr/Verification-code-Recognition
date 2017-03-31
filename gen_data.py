import random

infile=open('data.txt','r')
outfile=open('data_shuffle.txt','w')

list2= infile.readlines()
random.shuffle(list2)
print list2
for line in list2:
    print line
    outfile.write(line)

infile.close()
outfile.close()
