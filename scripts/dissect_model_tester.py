"""
This script is used to test the variations of dissect models. After creation of the model, 
it test against the actual test data.

"""
import nltk
import operator
import sys

from copy       import deepcopy
from lxml       import etree
from optparse   import OptionParser

from composes.transformation.scaling.ppmi_weighting                     import PpmiWeighting
from composes.transformation.dim_reduction.svd                          import Svd
from composes.transformation.feature_selection.top_feature_selection    import TopFeatureSelection
from composes.semantic_space.space                                      import Space
from composes.similarity.cos                                            import CosSimilarity
from composes.similarity.dot_prod                                       import DotProdSimilarity
from composes.similarity.lin                                            import LinSimilarity
from composes.utils                                                     import io_utils
from composes.utils                                                     import log_utils

from nltk.corpus    import wordnet as wn
from nltk.stem      import WordNetLemmatizer
from nltk.tag       import pos_tag
from nltk.tokenize  import word_tokenize

from SpaceSubClassModel import MySpace

###############################################################################
# This global model would store whatever final model we would chooose in the
# script.
final_model = None

# List of tags, only which we consider are important. Basically, all 
# other words dont add much to the context.
important_tags = ['NN', 'NNS', 'VB', 'VBP', 'VBN', 'VBG', 'VBD', 'VBZ', 'NNP', 'JJ', 'JJR', 'JJS', 'RB', 'N', 'PRP$', 'PRP']
###############################################################################

def train_core(rows_file, cols_file, sm_file, ppmi=False, top_features=None, svd=None, save_location=None):
    """
    Takes co-occurance relate files and train the model.
    
    @rows_file    : All the entries which label the rows of the matrix.
    @columss_file : All the entries which label the columns of the matrix.
    @sm_file      : All the co-occurance entries in the corpus.
    @ppmi         : Whether we want to do the Ppmi weighting.
    @TopFeatures  : To restrict the no of features are to be selected in total. 
                    None, signifies all the features have to be selected.
    @Svd          : If we want to reduce the dimensions. Not advised though.
                    None, signifies that dimensions have to be reduced.

    """
    global final_model

    core_space = MySpace.xbuild(data=sm_file, rows=rows_file, 
                                cols=cols_file, format="sm")

    if ppmi:
        core_space = core_space.apply(PpmiWeighting())
    
    if top_features:
        core_space = core_space.apply(TopFeatureSelection(int(top_features)))

    if svd:
        core_space = core_space.apply(Svd(int(svd)))
    
    final_model = core_space

    if save_location:
        io_utils.save(final_model, save_location)

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
    return tokens

###############################################################################
# Replacement Helpers.

def find_replacements_helper(imp_words, word, index, lwindow, rwindow, add):
    """
    This function actually runs the model and find replacements for the word.

    """
    left = index - lwindow if index - lwindow >= 0 else 0
    right = index + rwindow if index + rwindow <= len(imp_words) else len(imp_words)
    
    context_words = imp_words[left:index] + imp_words[index + 1:right]
    #print context_words, word
    # Gather all the context words in one vector.
    base_unison = None
    for x in context_words:
        try:
            if base_unison is None:
                base_unison = deepcopy(final_model.get_row(x))
            else:
                if add:
                    base_unison += final_model.get_row(x)
                else:
                    base_unison = base_unison.multiply(final_model.get_row(x))
        except KeyError, ex:
            print "Warning: " + x + " is not in entire corpus" 
            pass
    
    # Create a vector having context words and word to replace.
    if add:
        context_word_vector = base_unison + final_model.get_row(word) 
    else:
        context_word_vector = base_unison.multiply(final_model.get_row(word))
   
    results = {}
    cos_sim = CosSimilarity()
    for replacement, xx in final_model.get_neighbours(word, 75, cos_sim):
            # Ignore itself as a replacement.
            if replacement in context_words:
                continue
            # Get rid of cases like "fix" and "fixing".
            if word in replacement or replacement in word:
                continue
            # Replace only with the same POS tag.
            if replacement[-1] != word[-1]:
                continue
            
            replacement_vector = final_model.get_row(replacement)
            
            if add:
                context_repl_vector = base_unison + replacement_vector
            else:
                context_repl_vector = base_unison.multiply(replacement_vector)
            
            results[replacement] = cos_sim.get_sim(context_word_vector, context_repl_vector)

    return (word, map(lambda x: x[0][:-2], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]))

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

    #sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
    final_list = []
    for i, x in enumerate(imp_words):
        if x[0].startswith("_START_"):
            index = i
            x[0] = x[0][7:x[0].index("_END_")]
            final_list.append("_START_" + x[0].lower() + "_" + x[1][0].lower() + "_END_")
            word = word.lower() + "_" + x[1][0].lower()
        else:
            # Lemmatize all the words.
            word_postag = get_wordnet_pos(x[1])
            temp = x[0]
            if word_postag:
                temp = wnl.lemmatize(x[0], pos=word_postag)
            final_list.append(temp.lower() + "_" + x[1][0].lower())

    #print final_list
    try:
        return find_replacements_helper(final_list, word, index, int(lwindow), int(rwindow) + 1, add)
    except Exception:
        return "NONE"

###############################################################################

def get_options():
    parser = OptionParser()

    # Options related to the model.
    parser.add_option("-r", "--rows_file", dest="rows_file",
                      help="File containing rows of the corpus.")
    parser.add_option("-c", "--cols_file", dest="cols_file",
                      help="File containing columns of the corpus.")
    parser.add_option("-m", "--sm_file", dest="sm_file",
                      help="File containing co-occurance counts of the corpus.")
    parser.add_option("--ppmi", action="store_true", dest="ppmi",
                      help="If want to enable ppmi features.")
    parser.add_option("--top_features", dest="top_features",
                      help="Restrict number of features.")
    parser.add_option("--svd", dest="svd",
                      help="Dimension after reduction.")
    parser.add_option("--save_location", dest="save_location",
                      help="Save model to a pkl file, which can be re-used.")
    parser.add_option("--pkl_file", dest="pkl_file",
                      help="Location from where pkl file to read. Takes priority\
                           over other options.")

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
    
    if not opts.pkl_file:
        if not opts.rows_file or not opts.cols_file or not opts.sm_file:
            sys.exit("Please give either pkl file or combination of (row, col and sm file).")

    return (opts, args)

###############################################################################

if __name__ == "__main__":
    opts, args = get_options()

    # Either train or load the model.
    if not opts.pkl_file:
        train_core(opts.rows_file, opts.cols_file, opts.sm_file, opts.ppmi,
              opts.top_features, opts.svd, opts.save_location)
    else:
        final_model = io_utils.load(opts.pkl_file, MySpace)

    # Load and test the XMl input file.
    if not opts.xml_input:
        sys.exit("")

    # Read the file In.
    sxml = ""
    with open(opts.xml_input) as xml_file:
        sxml = xml_file.read()

    parser = etree.XMLParser(recover=True)
    tree   = etree.fromstring(sxml, parser=parser)

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
