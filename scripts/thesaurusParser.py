import sys

from urllib2 import urlopen

dictionary_template = "http://words.bighugelabs.com/api/2/c2b653776e7c1b3a05b7d16044028712/WORD/json"

input = open(sys.argv[1], "r")
output = open(sys.argv[2], "w")

results = {}
for line in input.readlines():
    url = dictionary_template.replace("WORD", line.strip())
    meaning = urlopen(url).readlines()
    results[line.strip()] = meaning

input.close()
output.write(str(results))
output.close()
