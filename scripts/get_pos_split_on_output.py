import sys

f = open(sys.argv[1], "r")
lines = f.readlines()
f.close()


f = open(sys.argv[2], "w")
for line in lines:
    if line.split(" ")[0][-1] == sys.argv[3]:
        f.write(line)
f.close()
    
