import sys
import mmap

file1 = sys.argv[1]
file2 = sys.argv[2]

rows_file = sys.argv[3] + "_rows"
cols_file = sys.argv[3] + "_cols"
sm_file   = sys.argv[3] + "_sm"

occur = {}

##############################################################
def dump(rows_file, cols_file, sm_file):
    global occur
    co_occur = occur

    # Write files.
    rows_file = open(rows_file, "w")
    cols_file = open(cols_file, "w")
    sm_file   = open(sm_file, "w")
    
    cols = {}
    for x in sorted(co_occur):
        for y in sorted(co_occur[x]):
            cols[y] = True
            sm_file.write(str(x) + "\t" + str(y) + "\t" + str(co_occur[x][y]) + "\n")
    
    sm_file.close()
    rows_file.write("\n".join(sorted(co_occur.keys())))
    cols_file.write("\n".join(sorted(cols.keys())))
    
    rows_file.close()
    cols_file.close()
    sm_file.close()
##############################################################

with open(file1, "r") as f:
    xmap = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) 
    for line in iter(xmap.readline, ""):
        line = line.strip().split("\t")
        word_1 = line[0]
        word_2 = line[1]
        count = int(line[2])
        if word_1 in occur:
            if word_2 in occur[word_1]:
                occur[word_1][word_2] += count
            else:
                occur[word_1][word_2] = count
        else:
            occur[word_1] = {word_2 : count}

with open(file2, "r") as f:
    xmap = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ) 
    for line in iter(xmap.readline, ""):
        line = line.strip().split("\t")
        word_1 = line[0]
        word_2 = line[1]
        count = int(line[2])
        if word_1 in occur:
            if word_2 in occur[word_1]:
                occur[word_1][word_2] += count
            else:
                occur[word_1][word_2] = count
        else:
            occur[word_1] = {word_2 : count}

dump(rows_file, cols_file, sm_file)
