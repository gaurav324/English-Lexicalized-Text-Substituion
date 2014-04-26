"""
This script is used to test the variations of dissect models. After creation of the model, 
it test against the actual test data.

Sample Invocation:

python /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/scripts/dissect_model_tester.py --pkl_file /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_with_ppmi.pkl --xml_input /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/TaskTestData/trial/lexsub_trial_cleaned.xml --lwindow 1 --rwindow 1 --ppmi --output_to nanda_out10

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

from BigThesaurus       import BigThesaurus
from SpaceSubClassModel import MySpace

###############################################################################
# This global model would store whatever final model we would chooose in the
# script.
final_model = None

# List of tags, only which we consider are important. Basically, all 
# other words dont add much to the context.
important_tags = ['NN', 'NNS', 'VB', 'VBP', 'VBN', 'VBG', 'VBD', 'VBZ', 'NNP', 'JJ', 'JJR', 'JJS', 'RB', 'N']

big_thesaurus = BigThesaurus()
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
        return [wn.ADJ, wn.ADJ_SAT]
    elif treebank_tag.startswith('V'):
        return [wn.VERB]
    elif treebank_tag.startswith('N'):
        return [wn.NOUN]
    elif treebank_tag.startswith('R'):
        return [wn.ADV]
    else:
        return [None]
    
def get_imp_words(tagged_sentence):
    """
    This function would filter words in the sentence whose
    tags don't belong to important tags list.

    """
    tokens = filter(lambda x: x != None, map(lambda x: x if x[1] in important_tags else None, tagged_sentence))
    return tokens

###############################################################################
# Replacement Helpers.

def find_replacements_helper(imp_words, word, index, lwindow, rwindow, 
                             add, enable_synset_avg, no_rerank, left_right_add,
                             thesaurus):
    """
    This function actually runs the model and find replacements for the word.

    """
    
    # Fetch left context words.
    #print imp_words, word, index, lwindow, rwindow, add, enable_synset_avg, no_rerank, left_right_add
    
    temp = index
    count = 0
    left_context_words = []
    while temp != 0 and count != lwindow:
        temp -= 1
        try:
            final_model.get_row(imp_words[temp])
        except KeyError, ex:
            print "Warning: " + imp_words[temp] + " is not in entire corpus" 
            continue
        count += 1
        left_context_words.append(imp_words[temp])

    # Fetch right context words.
    temp = index
    count = 0
    right_context_words = []
    while temp != len(imp_words) - 1 and count != rwindow:
        temp += 1
        try:
            final_model.get_row(imp_words[temp])
        except KeyError, ex:
            print "Warning: " + imp_words[temp] + " is not in entire corpus" 
            continue
    
        count += 1
        right_context_words.append(imp_words[temp])
    
    #print left_context_words, right_context_words

    # Gather all the context words in one vector.
    left_unison = None
    for x in left_context_words:
        if left_unison is None:
            left_unison = deepcopy(final_model.get_row(x))
        else:
            if add:
                left_unison += final_model.get_row(x)
            else:
                left_unison = left_unison.multiply(final_model.get_row(x))
    
    right_unison = None
    for x in right_context_words:
        if right_unison is None:
            right_unison = deepcopy(final_model.get_row(x))
        else:
            if add:
                right_unison += final_model.get_row(x)
            else:
                right_unison = right_unison.multiply(final_model.get_row(x))
    
    base_unison = None
    if left_unison is None:
        base_unison = right_unison
    elif right_unison is None:
        base_unison = left_unison
    else:
        if left_right_add or add:
            base_unison = left_unison + right_unison
        else:
            for x in right_context_words:
                left_unison = left_unison.multiply(final_model.get_row(x))

            base_unison = left_unison
    
    # Create a vector having context words and word to replace.
    if add:
        context_word_vector = base_unison + final_model.get_row(word) 
    else:
        context_word_vector = base_unison.multiply(final_model.get_row(word)) if base_unison is not None else final_model.get_row(word)

    results = {}
    cos_sim = CosSimilarity()

    #############################################################################
    # If we simply get the nearest neigbours of the actual context word.
    #############################################################################
    if no_rerank:
        results = final_model.get_xneighbours(context_word_vector, 10, cos_sim)
        return (word, map(lambda x: x[0][:-2], results))
    
    #############################################################################
    # Get the list of the similar words to the given vector.
    #############################################################################
    antonyms = big_thesaurus.antonyms(word)
    replacements = []
    if thesaurus > 0.0:
        synonyms = big_thesaurus.replacements(word)
        replacements = filter(lambda x: len(x.split(" ")) == 1, map(lambda x: x.lower() + "_" + word[-1], synonyms))

    how_many = int((1 - thesaurus) * len(replacements)) if thesaurus > 0.0 else 75
    if how_many > 0:
        replacements.extend(map(lambda x: x[0], final_model.get_neighbours(word, how_many, cos_sim)))
    for replacement in replacements:
            # Get rid of cases like "fix" and "fixing".
            if word[:-2] in replacement[:-2] or replacement[:-2] in word[:-2]:
                continue
            # Replace only with the same POS tag.
            if replacement[-1] != word[-1]:
                continue

            if antonyms is not None and replacement[:-2] in antonyms:
                continue
            
            try:
                replacement_vector = final_model.get_row(replacement)
            except Exception, ex:
                continue
            
            if add:
                context_repl_vector = base_unison + replacement_vector
            else:
                context_repl_vector = base_unison.multiply(replacement_vector)
            
            results[replacement] = cos_sim.get_sim(context_word_vector, context_repl_vector)
            wnl = WordNetLemmatizer()

            #############################################################################
            # This approach was to take similar words from similarity space and then
            # find synsets with the highest average replacement.
            #############################################################################
            if enable_synset_avg:
                synsets = wn.synsets(replacement[:-2])
                results_map = {}
                for synset in synsets:
                    postag_list = get_wordnet_pos(replacement[-1].upper())
                    if synset.pos in postag_list:
                        synset_syns = synset.lemma_names
                        avg = 0
                        count = 0
                        for syn in synset_syns:
                            if len(syn.split(" ")) > 1:
                                continue
                            if syn == replacement[:-2]:
                                continue
                            try:
                                replacement = str(syn) + "_" + replacement[-1]
                                replacement_vector = final_model.get_row(replacement)
                                if add:
                                    context_repl_vector = base_unison + replacement_vector
                                else:
                                    context_repl_vector = base_unison.multiply(replacement_vector)

                                simil = cos_sim.get_sim(context_word_vector, context_repl_vector)
                                avg += simil
                                count += 1
                            except:
                                pass
                        if count > 0:
                            avg /= count
                            results_map[synset] = avg
                #print replacement, results_map
                #print results
                if len(results_map.values()) > 0:
                    results[replacement] = max(results_map.values())
                else:
                    results[replacement] = 0.0
            #############################################################################

    #print results
    #print "###########################"
    #print context_words, word
    #print map(lambda x: x[0][:-2], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    #print "###########################"
    return (word, map(lambda x: x[0][:-2], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]))

def find_replacements(sentence, orig_word, lwindow, rwindow, add=False, 
                      enable_synset_avg=False, no_rerank=False, left_right_add=False,
                      thesaurus=0.0):
    """
    This function would be used to find replacements for the word present
    inside the sentence.

    @sentence          : Actual sentence in which word is present.
    @orig_word         : Word whose replacement is to be found
    @lwindow           : Number of context words in the left of the replacement.
    @rwindow           : Number of context words in the right of the replacement.
    @add               : Whether we are going to add the vectors. 
                         Otherwise default to multiply.
    @enable_synset_avg : Flag to switch synset avg approach.
    @no_rerank         : Flag to switch no_reranking approach.
    @left_right_add    : Multiply vectors in the left and right and add together.

    """
    # Remove the START and END temporarily and tag the data.
    word       = sentence[sentence.index('_START_') + 7 : sentence.index('_END_')]
    word_index = nltk.word_tokenize(sentence).index("_START_" + word + "_END_")
    t_sentence = sentence[:sentence.index('_START_')] + word + sentence[sentence.index('_END_') + 5:]

    # Tag the sentence and then bring the START and END back.
    tagged_sentence = nltk.pos_tag(nltk.word_tokenize(t_sentence))
    #print str(t_sentence), str(tagged_sentence)

    wnl = WordNetLemmatizer()
    tagged_sentence[word_index] = ["_START_" + word + "_END_", "NN"]
    
    # Remove all the words, whose tags are not important and also
    # get rid of smaller words.
    #print "********************************"
    #print tagged_sentence
    imp_words = filter(lambda x: len(x[0]) > 2, get_imp_words(tagged_sentence))
    #print imp_words
    #print "################################"

    final_list = []
    for i, x in enumerate(imp_words):
        if x[0].startswith("_START_"):
            index = i
            final_list.append("_START_" + orig_word + "_END_")
            word = orig_word
        else:
            # Lemmatize all the words.
            word_postag = get_wordnet_pos(x[1])[0]
            temp = x[0]
            if word_postag:
                temp = wnl.lemmatize(x[0], pos=word_postag)
            final_list.append(temp.lower() + "_" + x[1][0].lower())

    #print final_list
    try:
        return find_replacements_helper(final_list, word, index, int(lwindow),
                                        int(rwindow), add, enable_synset_avg, 
                                        no_rerank, left_right_add, thesaurus)
    except Exception, ex:
        print ex
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
                           over other options. Comma Semma separated list can \
                           also be provided.")

    # Options related to testing.
    parser.add_option("-x", "--xml_input", dest="xml_input",
                      help="Xml input file provided by the English Lexical\
                            Substituition task.")
    parser.add_option("--add", action="store_true", dest="add",
                      help="If we want to add vectors in context for finding replacements.")
    parser.add_option("--left_right_add", action="store_true", dest="left_right_add",
                      help="If we want to multiply left and right components and add them together.\
                            Only meaningful if only looking at one left and one right word.")
    parser.add_option("--lwindow", dest="lwindow", default=2,
                      help="Number of words to the left of the words to be replaced.")
    parser.add_option("--rwindow", dest="rwindow", default=2,
                      help="Number of words to the right of the words to be replaced.")
    parser.add_option("--output_to", dest="output_file",
                      help="File name where output has to be written.")
    parser.add_option("--enable_synset_avg", action="store_true", dest="enable_synset_avg",
                      help="If we want to improve results by avging over synsets.")
    parser.add_option("--no_rerank", action="store_true", dest="no_rerank",
                      help="Instead of getting all the similar words and re-ranking \
                            them, try creating a vector and find similar words to \
                            that.")
    parser.add_option("--thesaurus", dest="thesaurus", default=0,
                      help="By default, it would be zero. If 0.5, we would extract get \
                            equal number of words from similarity model and rank them.")
    
    opts, args = parser.parse_args()
    
    if not opts.pkl_file:
        if not opts.rows_file or not opts.cols_file or not opts.sm_file:
            sys.exit("Please give either pkl file or combination of (row, col and sm file).")
    
    if opts.add and opts.left_right_add:
        sys.exit("Either give add or left_right_add. You cannot have them together.")

    if opts.left_right_add and (opts.lwindow != "1" or opts.rwindow != "1"):
        sys.exit("Left and right windows should just be one, when adding left and right.")

    return (opts, args)

###############################################################################

if __name__ == "__main__":
    opts, args = get_options()

    # Either train or load the model.
    if not opts.pkl_file:
        train_core(opts.rows_file, opts.cols_file, opts.sm_file, opts.ppmi,
              opts.top_features, opts.svd, opts.save_location)
    else:
        core_space = io_utils.load(opts.pkl_file, MySpace)

        if opts.ppmi:
            core_space = core_space.apply(PpmiWeighting())
    
        if opts.top_features:
            core_space = core_space.apply(TopFeatureSelection(int(opts.top_features)))
    
        if opts.svd:
            core_space = core_space.apply(Svd(int(opts.svd)))
        
        final_model = core_space

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
                
                #print sentence, word, index
                word = str(el.items()[0][1])
                
                # Some of the words have WORD.a.n type of suffix.
                if word[:-2][-2] == ".":
                    word = word[:-4] + "_" + word[-1]
                else:
                    word = word[:-2] + "_" + word[-1]
                if word[-1] == "a":
                    word = word[:-1] + "j"
                word = word.lower()

                result = find_replacements(sentence, word, opts.lwindow, opts.rwindow, opts.add, 
                                           opts.enable_synset_avg, opts.no_rerank, opts.left_right_add,
                                           float(opts.thesaurus))
                values = ";".join(result[1])
                #print sentence, result
                f.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " ::: " + values)
                f.write("\n")
    f.close()
    print "Output file written."
