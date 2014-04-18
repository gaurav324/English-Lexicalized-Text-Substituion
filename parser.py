import re
import sys, os

allowed_pos = ['nn', 'nns', 'vb', 'vbp', 'vbg', 'vbd', 'nnp', 'jj', 'jjr', 'rb', 'n', 'prp$', 'prp', 'wdt']
co_occur = {}

regex = re.compile(r'^[a-zA-Z]+$')

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
        
        cleaned = clean(current_word[0])
        if not cleaned:
            continue
        else:
            current_word[0] = cleaned.string
        try:
            if current_word[2] not in allowed_pos:
                continue
        except IndexError, ex:
            print "syapa (cant access 2 index): ", current_word
        window = 2; window_t = 0
        for j in range(count - i - 1):
            if (window_t == window):
                break
            else:
                next_word = sentence[i + j + 1]
                if next_word.strip().startswith("<text") or next_word.strip().startswith("</text"):
                    continue
                next_word = next_word.split("\t")
                cleaned = clean(next_word[0])
                if not cleaned:
                    continue
                else:
                    next_word[0] = cleaned.string
                #print i, j, next_word
                try:
                    if next_word[2] not in allowed_pos:
                        continue
                    else:
                        window_t += 1
                        if current_word[0] in co_occur:
                            if next_word[0] in co_occur[current_word[0]]:
                                co_occur[current_word[0]][next_word[0]] += 1
                            else:
                                co_occur[current_word[0]] = {next_word[0] : 1}
                        else:
                            co_occur[current_word[0]] = {next_word[0] : 1}
                except IndexError, ex:
                    print "syapa: ", ex
                
        
files = os.listdir(sys.argv[1]) if os.path.isdir(sys.argv[1]) else [sys.argv[1]]

for file in files:
    with open(file, "r")  as f:
        sentence = []
        for line in f:
            if line.strip().startswith("<s>"):
                sentence = []
            elif line.strip().startswith("<text") or line.strip().startswith("</text"):
                #print "Going in: " + line.strip()
                continue
            elif line.strip().startswith("</s>"):
                update_co_occur(sentence)
                continue
            else:
                if line.strip() != "":
                    sentence.append(line.strip())

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
