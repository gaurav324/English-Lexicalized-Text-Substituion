"""
This script is used to test the variations of Google Binary N gram models. After creation of the model, it test against the actual test data.

Sample Invocation:

python google_model_tester.py --bin_file GoogleNews-vectors-negative300.bin --xml_input TaskTestData/trial/lexsub_trial_cleaned.xml --lwindow 1 --rwindow 1 --output_to nanda_out10

"""
import nltk
import operator
import sys

from lxml       import etree
from optparse   import OptionParser

from copy 		import deepcopy
from gensim 		import 	utils, matutils
from gensim.models 	import word2vec

from nltk.corpus 	import wordnet as wn
from nltk.tag 		import pos_tag
from nltk.tokenize 	import word_tokenize
from nltk.stem      	import WordNetLemmatizer

from numpy 		import dot, multiply, add

###############################################################################
# This global model would store whatever final model we would chooose in the script.

final_model = None

# List of tags, only which we consider are important. All other words dont add much to the context.
important_tags = ['NN', 'NNS', 'VB', 'VBP', 'VBN', 'VBG', 'VBD', 'VBZ', 'NNP', 'JJ', 'JJR', 'JJS', 'RB', 'N', 'PRP$', 'PRP']

###############################################################################
# Helper functions.

def get_wordnet_pos(treebank_tag):
    """
    This function would return corresponding wordnet POS tag for 
    penn-tree bank POS tag.

    """
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None
    
def get_imp_words(tagged_sentence):
    """
    This function would filter words in the sentence whose
    tags don't belong to important tags list.

    """
    tokens = filter(lambda x: x != None, map(lambda x: x if x[1] in important_tags else None, tagged_sentence))
    #print tokens
    return tokens

def cosine(ww1, ww2):
    return dot(matutils.unitvec(ww1), matutils.unitvec(ww2))

###############################################################################
# Replacement Helpers.

def find_replacements_helper(imp_words, word, index, lwindow, rwindow, add):
    """
    This function actually runs the model and find replacements for the word.

    """
    #print word + " " + imp_words + " " + add

    left = index - lwindow if index - lwindow >= 0 else 0
    right = index + rwindow if index + rwindow <= len(imp_words) else len(imp_words)
    
    context_words = imp_words[left:index] + imp_words[index + 1:right]
    #print context_words, word
    # Gather all the context words in one vector.

    base_unison = None
    for x in context_words:
        try:
            if base_unison is None:
                base_unison = deepcopy(final_model.syn0[final_model.vocab[x].index])
            else:
                if add:
                    base_unison += final_model.syn0[final_model.vocab[x].index]
                else:
                    base_unison = multiply(base_unison, final_model.syn0[final_model.vocab[x].index])
        except KeyError, ex:
            print "Warning: " + x + " is not in entire corpus" 
            pass
    
    # Create a vector having context words and word to replace.
    if add:
        context_word_vector = base_unison + final_model.syn0[final_model.vocab[word].index]
    else:
        context_word_vector = multiply(base_unison, final_model.syn0[final_model.vocab[word].index])
   
    results = {}
    for replacement, xx in final_model.most_similar(word,  topn=30):
            # Ignore itself as a replacement.
            if replacement in context_words:
                continue
            # Get rid of cases like "fix" and "fixing".
            if word in replacement or replacement in word:
                continue
            # Replace only with the same POS tag.
            if replacement[-1] != word[-1]:
                continue
            
            replacement_vector = final_model.syn0[final_model.vocab[replacement].index]
            
            if add:
                context_repl_vector = base_unison + replacement_vector
            else:
                context_repl_vector = multiply(base_unison, replacement_vector)

            results[replacement] = cosine(context_word_vector, context_repl_vector) 
            #print context_word_vector +" " + context_repl_vector
    #print results        
    return (word, map(lambda x: x[0], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]))

###############################################################################
def find_replacements(sentence, lwindow, rwindow, add=False):
    """
    This function would be used to find replacements for the word present
    inside the sentence.

    @sentence: Actual sentence in which word is present.
    @lwindow : Number of context words in the left of the replacement.
    @rwindow : Number of context words in the right of the replacement.
    @add     : Whether we are going to add the vectors. 
               Otherwise default to multiply.

    """
    # Remove the START and END temporarily and tag the data.
    word       = sentence[sentence.index('_START_') + 7 : sentence.index('_END_')]
    word_index = nltk.word_tokenize(sentence).index("_START_" + word + "_END_")
    t_sentence = sentence[:sentence.index('_START_')] + word + sentence[sentence.index('_END_') + 5:]

    # Tag the sentence and then bring the START and END back.
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(t_sentence))
    #print sentence, tagged_sentence

    wnl = WordNetLemmatizer()
    word_postag = get_wordnet_pos(tagged_sentence[word_index][1])
    if word_postag:
        word = wnl.lemmatize(word, pos=word_postag)
    tagged_sentence[word_index] = ["_START_" + word + "_END_", tagged_sentence[word_index][1]]
    
    # Remove all the words, whose tags are not important and also
    # get rid of smaller words.
    imp_words = filter(lambda x: len(x[0]) > 2, get_imp_words(tagged_sentence))
    #print imp_words

    final_list = []
    for i, x in enumerate(imp_words):
        if x[0].startswith("_START_"):
            index = i
            x[0] = x[0][7:x[0].index("_END_")]
            final_list.append("_START_" + x[0].lower() + "_" + x[1][0].lower() + "_END_")
            word = word.lower() #+ "_" + x[1][0].lower()
            #print word
        else:
            # Lemmatize all the words.
            word_postag = get_wordnet_pos(x[1])
            temp = x[0]
            if word_postag:
                temp = wnl.lemmatize(x[0], pos=word_postag)
            final_list.append(temp.lower()) # + "_" + x[1][0].lower())

    try:
        return find_replacements_helper(final_list, word, index, int(lwindow), int(rwindow) + 1, add)
    except Exception:
        return "NONE"

###############################################################################

def get_options():
    parser = OptionParser()

    # Options related to the model.
    parser.add_option("--bin_file", dest="bin_file",
                      help="Location from where bin file is to be read.")

    # Options related to testing.
    parser.add_option("-x", "--xml_input", dest="xml_input",
                      help="Xml input file provided by the English Lexical\
                            Substituition task.")
    parser.add_option("--add", action="store_true", dest="add",
                      help="If we want to add vectors in context for finding replacements.")
    parser.add_option("--lwindow", dest="lwindow", default=2,
                      help="Number of words to the left of the words to be replaced.")
    parser.add_option("--rwindow", dest="rwindow", default=2,
                      help="Number of words to the right of the words to be replaced.")
    parser.add_option("--output_to", dest="output_file",
                      help="File name where output has to be written.")
    
    opts, args = parser.parse_args()
    
    if not opts.bin_file:
        sys.exit("Please give either bin file.")

    return (opts, args)

###############################################################################

if __name__ == "__main__":
    opts, args = get_options()

    # Load and test the XMl input file.
    if not opts.xml_input:
        sys.exit("")

    # Read the file In.
    sxml = ""
    with open(opts.xml_input) as xml_file:
        sxml = xml_file.read()

    parser = etree.XMLParser(recover=True)
    tree   = etree.fromstring(sxml, parser=parser)

    # Either train or load the model.
    final_model = word2vec.Word2Vec.load_word2vec_format(opts.bin_file, binary=1)

    f = open(opts.output_file, "w")
    for el in tree.findall('lexelt'):
        for ch in el.getchildren():
            for ch1 in ch.getchildren():
                sentence = ch1.text
                
                #word     = sentence[sentence.index('_START_') + 7 : sentence.index('_END_')]
                #index    = nltk.word_tokenize(sentence).index("_START_" + word + "_END_")
                #sentence = sentence[:sentence.index('_START_')] + word + sentence[sentence.index('_END_') + 5:]
                
                #print sentence, word, index
                result = find_replacements(sentence, opts.lwindow, opts.rwindow, opts.add)
                values = ";".join(result[1])
                f.write(str(result[0].replace("_", ".")) + " " + str(ch.items()[0][1]) + " :: " + values)
                f.write("\n")
    f.close()
    print "Output file written."
