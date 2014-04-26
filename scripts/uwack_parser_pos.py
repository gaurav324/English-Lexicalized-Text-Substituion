# Sample Invocation
# python uwack_parser_pos.py test_dir rows cols matrix

import enchant
import re
import sys
import os
import mmap

##############################################################
# Globals.

allowed_pos = ['jj','jjr','jjs','jjt','nn','nns','np','nps','nr','nrs','rb','rbr','rbs','vb','vbd','vbg','vbn','vbz', 'vbp','vhd','vhp','vhz','vv','vvd','vvg','vvn','vvp','vvz', 'nnp', 'jj$','jjr$','jjs$','jjt$','nn$','np$','nps$','rb$','vb$','vbd$','vbn$','vbz$','nnp$']

co_occur = {}

# Regex to match only words having letters. Filter out numbers, crappy urls etc.
regex = re.compile(r'^[a-zA-Z]+$')

english_dict = enchant.Dict("en_US")
##############################################################
# Helper Functions.

__cleaned__ = {}
# Clean the word.
def clean(word):
    if word in __cleaned__:
        return __cleaned__[word]

    xword = word.strip()
    if xword.endswith("."):
        xword = word[:-1]
    if (xword.strip() == "" or len(xword) < 2):
        __cleaned__[word] = None
        return None

    try:
        xword = unicode(xword, "utf-8")
        if (english_dict.check(xword) or "-" in xword):
            result = regex.match(xword)
            __cleaned__[word] = result
            return result
    except:
        pass
    __cleaned__[word] = None
    return None

# Sentence is a collection of the words tagged. Sentence is a list and each word is stored as string.
def update_co_occur(sentence):
    global co_occur, allowed_pos
    sentence = map(lambda x: x.lower(), sentence)
    
    #print sentence
    count = len(sentence)
    for i in range(count):
        current_word = sentence[i].split("\t")
        
        # Clean the current word. If it is not clean, ignore this and proceed.
        cleaned = clean(current_word[1])
        if not cleaned:
            continue
        else:
            current_word[1] = cleaned.string

        # Check the Pos Tag is present in the list of allowed pos_tags.
        try:
            if current_word[2] not in allowed_pos:
                continue
        except IndexError, ex:
            print "Error-Location1: ", current_word
	
    	#if current_word[1].endswith("."):
    	#    current_word[1] = current_word[1][:-1]	
	current_word[1] = current_word[1] + '_' + current_word[2][0]

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
                cleaned = clean(next_word[1])
                if not cleaned:
                    continue
                else:
                    next_word[1] = cleaned.string

                try:
                    if next_word[2] not in allowed_pos:
                        continue
                    else:
                        # Update count, telling how many words in the window have been included.
			
        		#if next_word[1].endswith("."):
            		#    next_word[1] = next_word[1][:-1]
			next_word[1] = next_word[1] + '_' + next_word[2][0]
                        window_t += 1
                        
                        # Update co-occurance matrix.
                        if current_word[1] in co_occur:
                            if next_word[1] in co_occur[current_word[1]]:
                                co_occur[current_word[1]][next_word[1]] += 1
                            else:
                                co_occur[current_word[1]][next_word[1]] = 1
                        else:
                            co_occur[current_word[1]] = {next_word[1] : 1}
			    if ((len(co_occur) % 10000) == 0):
			        print len(co_occur) 
                except IndexError, ex:
                    print "Error-Location2: ", ex

##############################################################
def dump(prefix):
    # Write files.
    rows_file = open(str(prefix) + "_" + sys.argv[2], "w")
    cols_file = open(str(prefix) + "_" + sys.argv[3], "w")
    sm_file   = open(str(prefix) + "_" + sys.argv[4], "w")

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
# This is where main action starts.

count = 0
# If first argument is folder, get list of files inside it. If it is a file, then use that file.
files = map(lambda x: sys.argv[1] + "/" + x, os.listdir(sys.argv[1])) if os.path.isdir(sys.argv[1]) else [sys.argv[1]]
#files = os.listdir(sys.argv[1]) if os.path.isdir(sys.argv[1]) else [sys.argv[1]]
for file in files:
    count += 1
    with open(file, "r")  as f:
        sentence = []
        xmap = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
	for line in iter(xmap.readline, ""):
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
    dump(count)
#    if (count % 20 == 0):
#	dump(count)
#if(count % 20 !=0):
#    dump(count)

# End of File.
##############################################################

