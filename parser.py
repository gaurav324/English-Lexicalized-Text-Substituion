# Sample Invocation
# python parser.py test_10000 test10000_rows test10000_cols test10000_sm

import re
import sys, os

##############################################################
# Globals.

allowed_pos = ['nn', 'nns', 'vb', 'vbp', 'vbg', 'vbd', 'nnp', 'jj', 'jjr', 'rb', 'n', 'prp$', 'prp', 'wdt']
co_occur = {}

# Regex to match only words having letters. Filter out numbers, crappy urls etc.
regex = re.compile(r'^[a-zA-Z]+$')

##############################################################
# Helper Functions.

# Clean the word. 
def clean(word):
    if word.endswith("."):
        word = word[:-1]
    return regex.match(word)

# Sentence is a collection of the words tagged. Sentence is a list and each word is stored as string.
def update_co_occur(sentence):
    global co_occur, allowed_pos
    sentence = map(lambda x: x.lower(), sentence)
    
    #print sentence
    count = len(sentence)
    for i in range(count):
        current_word = sentence[i].split("\t")
        
        # Clean the current word. If it is not clean, ignore this and proceed.
        cleaned = clean(current_word[0])
        if not cleaned:
            continue
        else:
            current_word[0] = cleaned.string

        # Check the Pos Tag is present in the list of allowed pos_tags.
        try:
            if current_word[2] not in allowed_pos:
                continue
        except IndexError, ex:
            print "Error-Location1: ", current_word

        # Window size is fixed to be 2. This signifies that find next two words, which have their
        # POS tags in the allowed list. If EOS comes first, then don't proceed further.
        window = 2; window_t = 0
        for j in range(count - i - 1):
            if (window_t == window):
                break
            else:
                next_word = sentence[i + j + 1]
                if next_word.strip().startswith("<text") or next_word.strip().startswith("</text"):
                    continue
                next_word = next_word.split("\t")

                # Clean the next word. If next word is dirty enough, then simply ignore and continue.
                cleaned = clean(next_word[0])
                if not cleaned:
                    continue
                else:
                    next_word[0] = cleaned.string

                try:
                    if next_word[2] not in allowed_pos:
                        continue
                    else:
                        # Update count, telling how many words in the window have been included.
                        window_t += 1
                        
                        # Update co-occurance matrix.
                        if current_word[0] in co_occur:
                            if next_word[0] in co_occur[current_word[0]]:
                                co_occur[current_word[0]][next_word[0]] += 1
                            else:
                                co_occur[current_word[0]] = {next_word[0] : 1}
                        else:
                            co_occur[current_word[0]] = {next_word[0] : 1}
                except IndexError, ex:
                    print "Error-Location2: ", ex

##############################################################
# This is where main action starts.

# If first argument is folder, get list of files inside it. If it is a file, then use that file.
files = os.listdir(sys.argv[1]) if os.path.isdir(sys.argv[1]) else [sys.argv[1]]
for file in files:
    with open(file, "r")  as f:
        sentence = []
        for line in f:
            if line.strip().startswith("<s>"):
                sentence = []
            elif line.strip().startswith("<text") or line.strip().startswith("</text"):
                continue
            elif line.strip().startswith("</s>"):
                update_co_occur(sentence)
                continue
            else:
                if line.strip() != "":
                    sentence.append(line.strip())

# Write files.
rows_file = open(sys.argv[2], "w")
cols_file = open(sys.argv[3], "w")
sm_file = open(sys.argv[4], "w")

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

# End of File.
##############################################################

