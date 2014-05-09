import sys

f = open(sys.argv[1], "r")
lines = f.readlines()
f.close()

from BigThesaurus import BigThesaurus
thes = BigThesaurus()

total = 0
not_found = 0

seen = {}
for line in lines:
    if line.strip() == "":
        continue
    words = line.split("::")[1].split(";")
    #print line.split(" ")[0][:-2] + "_" + line.split(" ")[0][-1]
    replacements = thes.replacements(line.split(" ")[0][:-2] + "_" + line.split(" ")[0][-1])
    #print replacements
    for word in words:
        word = word[:-2].strip()
        if word is None or word == "":
            continue
        if word not in seen:
            seen[word] = True
            #print word, replacements
            if len(word.split(" ")) > 1:
                continue
            if word not in replacements:
                #print word, "not found."
                #print "\n"
                not_found += 1
                #print "Found: ", word
            total += 1

print "Total: ", total, "Not Found: ", not_found
