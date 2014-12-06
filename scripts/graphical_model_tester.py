"""
This script is used to test the variations of dissect models. After creation of the model, 
it test against the actual test data.

Sample Invocation:

python /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/scripts/dissect_model_tester.py --pkl_file /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_with_ppmi.pkl --xml_input /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/TaskTestData/trial/lexsub_trial_cleaned.xml --lwindow 1 --rwindow 1 --ppmi --output_to nanda_out10

"""
import copy
import nltk

import numpy as np
import opengm
import operator
import sys


from copy       import deepcopy
from lxml       import etree
from math       import sqrt, log
from numpy      import zeros, array
from optparse   import OptionParser

from composes.transformation.scaling.ppmi_weighting                     import PpmiWeighting
from composes.transformation.scaling.plmi_weighting                     import PlmiWeighting
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

cos_sim = CosSimilarity()
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
# Find the Pos tag nearest to index word in imp_words

def find_pos_word_left(imp_words, pos_tag, index, window):
    context_words = []
    temp = index
    count = 0
    while temp > 0 and count < window:
        temp -= 1
        if (imp_words[temp][-1] == str(pos_tag)):
            try:
                final_model.get_row(imp_words[temp])
                count += 1
                context_words.append(imp_words[temp])
            except KeyError, ex:
                print "Warning: " + imp_words[temp] + " is not in entire corpus"
            continue

    return context_words

def find_pos_word_right(imp_words, pos_tag, index, window):
    context_words = []
    temp = index
    count = 0
    while temp < len(imp_words) - 1 and count < window:
        temp +=1
        if (imp_words[temp][-1] == str(pos_tag)):
            try:
                final_model.get_row(imp_words[temp])
                count += 1
                context_words.append(imp_words[temp])
            except KeyError, ex:
                print "Warning: " + imp_words[temp] + " is not in entire corpus"
            continue

    return context_words

###############################################################################
# Get replacements for the given word from the dictionary.
def replacements(word):
    replacements = []
    synonyms = big_thesaurus.replacements(word)
    replacements = filter(lambda x: len(x.split(" ")) == 1, map(lambda x: x.lower() + "_" + word[-1], synonyms))
    ans = []
    for replacement in replacements:
        ans.append(replacement)
    return ans

###############################################################################
# Get unary factors for the observed node.
def get_unary_factors(observed_node):
    paraphrase_words = replacements(observed_node)
    ans = {}
    observed_vector = final_model.get_row(observed_node)
    for word in paraphrase_words:
        try:
            word_vector = final_model.get_row(word)
        except Exception, ex:
            continue
        score = cos_sim.get_sim(observed_vector, word_vector)
        ans[word] = score
    return ans
        
###############################################################################
# Main function for generating graphical model from the given set of words.
# Model would be generated with words stringed together from left to right.
def getGraphicalModel(words):
    #print 1
    noNodes = sum(map(lambda x : 1 if x is not None else 0, words))
    
    word_factors_list = []
    no_of_states = []
    for word in words:
        if word is not None:
            factors = get_unary_factors(word)
            word_factors_list.append(factors)
            no_of_states.append(len(factors.keys()))
        
    #print 2
    gm = opengm.graphicalModel(no_of_states)
    
    # Add unary factor nodes for each word factor.
    for i, word_factors in enumerate(word_factors_list):
        factor_handle = gm.addFunction(np.array(word_factors.values()))
        gm.addFactor(factor_handle, i)
    
    #print 3
    # TODO: Assuming that relation exists only left to right
    for i in range(len(word_factors_list) - 1):
        # TODO: Just getting the similarity score.
        words_i = word_factors_list[i].keys()
        words_i1 = word_factors_list[i + 1].keys()
        
        binary_func = []
        for word_a in words_i:
            word_a_values = []
            for word_b in words_i1:
                word_a_values.append(cos_sim.get_sim(final_model.get_row(word_a), final_model.get_row(word_b)))
            binary_func.append(word_a_values)
        factor_handle = gm.addFunction(np.array(binary_func))
        gm.addFactor(factor_handle, [i, i+1])
    #print 4
    #opengm.visualizeGm(gm)
    inf = opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(damping=0.05))
    inf.infer()
    #print 5
    return inf
###############################################################################
# We don't want to create a vector over the entire english language.
# Therefore we create a vector over the parent word and its replacements's replacements.
def getWordStateLength(word):
    replacements = get_unary_factors(word)
    word_total_vector = []
    xreplacements = []
    for replacement in replacements:
        new_replacements = get_unary_factors(replacement)
        if len(new_replacements.keys()) > 0:
            xreplacements.append(replacement)
            word_total_vector.append(replacement)
            word_total_vector.extend(new_replacements.keys())
        
    ans = list(set(word_total_vector))
    final_ans = {}
    for x in ans:
        final_ans[x] = 0
        
    return xreplacements, final_ans

###############################################################################
# Given collection of words and the index for which we are calculating marginals.
def computeVectorsForJenson(words, index):
    # Index signifies the index of the interest, for which we need to find the best replacement.
    #print "Words are: ", words
    #replacements = get_unary_factors(words[index]).keys()
    replacements, wordStateMap = getWordStateLength(words[index])
    
    jenson_vectors = []
    
    inf = getGraphicalModel(words)
    #print "model constructed"

    marginals = inf.marginals(index)
    #print marginals, replacements
    local_map = copy.deepcopy(wordStateMap)

    for i, word in enumerate(replacements):
        local_map[word] = marginals[0][i]
    jenson_vectors.append(local_map)
        
    for replacement in replacements:
        new_words = copy.deepcopy(words)
        new_words[index] = replacement
       
        #print "New words model", new_words 
        inf = getGraphicalModel(new_words)
        #print "New model constructed"
        marginals = inf.marginals(index)
        local_map = copy.deepcopy(wordStateMap)
        xreplacements = get_unary_factors(replacement).keys()
        for i, word in enumerate(xreplacements):
            local_map[word] = marginals[0][i]
        jenson_vectors.append(local_map)
       
    #print "models for replacements also done." 
    return replacements, jenson_vectors 

###############################################################################
class JSD(object):
    def __init__(self):
        self.log2 = log(2)


    def KL_divergence(self, p, q):
        """ Compute KL divergence of two vectors, K(p || q)."""
        return sum(p[x] * log((p[x]) / (q[x])) for x in range(len(p)) if p[x] != 0.0 or p[x] != 0)

    def Jensen_Shannon_divergence(self, p, q):
        """ Returns the Jensen-Shannon divergence. """
        self.JSD = 0.0
        weight = 0.5
        average = zeros(len(p)) #Average
        for x in range(len(p)):
            average[x] = weight * p[x] + (1 - weight) * q[x]
            self.JSD = (weight * self.KL_divergence(array(p), average)) + ((1 - weight) * self.KL_divergence(array(q), average))
        return 1-(self.JSD/sqrt(2 * self.log2))

###############################################################################
# Replacement Helpers.
def find_replacements_helper(imp_words, word, index, lwindow, rwindow,
                             add, enable_synset_avg, no_rerank, left_right_add,
                             thesaurus, pos_context):
    """
    This function actually runs the model and find replacements for the word.

    """

    # Fetch left context words.
    #print pos_context, imp_words, word, index, lwindow, rwindow, add, enable_synset_avg, no_rerank, left_right_add
    left_context_words = []
    right_context_words = []
    if pos_context:
        if (word[-1] == 'j'):
            # Adjective

            left_context_words  = []
            right_context_words = find_pos_word_right(imp_words, "n", index, rwindow) 
        
        elif(word[-1] == 'n'):
            # Noun :- Base_unions = word * ajective to left + word * verb to left + word * verb to right 

            left_context_words = find_pos_word_left(imp_words, "j", index, lwindow) 
            left_context_words.extend(find_pos_word_left(imp_words, "v", index, lwindow))

            right_context_words = find_pos_word_right(imp_words, "v", index, rwindow) 
        
        elif (word[-1] == 'r'):
            #Adverb.

            # Option-1.
            # (a) BASE_VECTOR = T * V2L + T * V2R
            #left_context_words  = find_pos_word_left(imp_words, "v", index, lwindow) 
            #right_context_words = find_pos_word_right(imp_words, "v", index, rwindow) 

            # Option-2.
            # (b) BASE_VECTOR = T * V2L + T * N2L + T * V2R + T * N2R
            left_context_words.extend(find_pos_word_left(imp_words, "n", index, lwindow))
            right_context_words.extend(find_pos_word_left(imp_words, "n", index, lwindow))

        elif (word[-1] == 'v'):
            # Verb

            # Option - 1
            # (a) BASE_VECTOR = T * A2L + T * N2R
            #left_context_words  = find_pos_word_left(imp_words, "j", index, lwindow) 
            #right_context_words = find_pos_word_right(imp_words, "n", index, rwindow) 
            #(c) BASE_VECTOR = T * A2L + T * N2L + T * A2R + T * N2R
            #left_context_words  = find_pos_word_left(imp_words, "j", index, lwindow)
            #left_context_words  = find_pos_word_left(imp_words, "n", index, lwindow)
            #right_context_words = find_pos_word_right(imp_words, "j", index, rwindow)
            #right_context_words = find_pos_word_right(imp_words, "n", index, rwindow)

            #(b) BASE_VECTOR = T * A2L + T * N2L + T * A2R + T * N2R + T* D2L + T * D2R
            left_context_words  = find_pos_word_left(imp_words, "j", index, lwindow)
            left_context_words.extend(find_pos_word_left(imp_words, "n", index, lwindow))
            left_context_words.extend(find_pos_word_left(imp_words, "r", index, lwindow))
            right_context_words = find_pos_word_right(imp_words, "r", index, rwindow)
            right_context_words.extend(find_pos_word_right(imp_words, "j", index, rwindow))
            right_context_words.extend(find_pos_word_right(imp_words, "n", index, rwindow))

        #(d) BASE_VECTOR = T * A2L + T * N2R + T * D2L + T * D2R
            #left_context_words  = find_pos_word_left(imp_words, "j", index, lwindow)
            #left_context_words  = find_pos_word_left(imp_words, "r", index, lwindow)
            #right_context_words = find_pos_word_right(imp_words, "r", index, rwindow)
            #right_context_words = find_pos_word_right(imp_words, "n", index, rwindow)

    if (pos_context is None):
        temp = index
        count = 0
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
            if add or pos_context:
                left_unison += final_model.get_row(x)
            else:
                left_unison = left_unison.multiply(final_model.get_row(x))
    #print "One"
    right_unison = None
    for x in right_context_words:
        if right_unison is None:
            right_unison = deepcopy(final_model.get_row(x))
        else:
            if add or pos_context:
                right_unison += final_model.get_row(x)
            else:
                right_unison = right_unison.multiply(final_model.get_row(x))

    #print "Two"
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

    #print "Three"
    # Create a vector having context words and word to replace.
    if add:
        context_word_vector = base_unison + final_model.get_row(word)
    else:
        context_word_vector = base_unison.multiply(final_model.get_row(word)) if base_unison is not None else final_model.get_row(word)

    #print "Four"
    results = {}
    cos_sim = CosSimilarity()
    
    ############################################################################
    # Graphical Model representation.
    words = []
    for x in left_context_words:
        if x[:-2] in big_thesaurus._data_ and len(big_thesaurus.replacements(x)) > 0:
            words.append(x)
        else:
            print "Ignored: ", x
    index = len(words)
    words.append(word)
    for x in right_context_words:
        if x[:-2] in big_thesaurus._data_ and len(big_thesaurus.replacements(x)) > 0:
            words.append(x)
        else:
            print "Ignored: ", x
    
    replacements, vectors = computeVectorsForJenson(words, index)
    jsd = JSD()
    results = {}
    for i, vector in enumerate(vectors):
        if i == 0:
            continue
        divg = jsd.Jensen_Shannon_divergence(vectors[0].values(), vectors[i].values())
        #print "Divg returned"
        results[replacements[i-1]] = -1 * divg 
    print "Jensen divergene calculated."
    
    results = map(lambda x: x[0][:-2], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    if (len(results) < 10):
        for repl in  big_thesaurus.replacements(word):
            if repl not in results:
                results.append(repl)
                if len(results) >= 10:
                    break

    return (word, results) 
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
                if (word[-1] != "n"):
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
                context_repl_vector = base_unison.multiply(replacement_vector) if base_unison is not None else replacement_vector


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
                                    context_repl_vector = base_unison.multiply(replacement_vector) if base_unison is not None else replacement_vector
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
    #print left_context_words,right_context_words, word
    #print map(lambda x: x[0][:-2], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    #print "###########################"
    results = map(lambda x: x[0][:-2], sorted(results.iteritems(), key=operator.itemgetter(1), reverse=True)[:10])
    if (len(results) < 10):
        for repl in  big_thesaurus.replacements(word):
            if repl not in results:
                results.append(repl)
                if len(results) >= 10:
                    break
        
    return (word, results)

def find_replacements(sentence, orig_word, lwindow, rwindow, add=False, 
                      enable_synset_avg=False, no_rerank=False, left_right_add=False,
                      thesaurus=0.0, pos_context=False):
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
    @thesaurus         : Give what ratio of candidates are to be taken from thesaurus.
    @pos_context       : Boolean which tells us whether to switch on semantic similarity

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
                                        no_rerank, left_right_add, thesaurus, 
                                        pos_context)
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
    parser.add_option("--plmi", action="store_true", dest="plmi",
                      help="If want to enable plmi features.")
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
    parser.add_option("--pos_context", action="store_true", dest="pos_context",
                      help="To include the semantic context suggested by Gemma.")
    parser.add_option("--start_word", dest="start_word",)
    parser.add_option("--end_word", dest="end_word",)
 
    opts, args = parser.parse_args()
   
    if opts.plmi and opts.ppmi:
        sys.exit("Choose one among ppmi and plmi")

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
        elif opts.plmi:
            core_space = core_space.apply(PlmiWeighting())
    
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
    f_n = open(opts.output_file + "__NOUN__", "w")
    f_a = open(opts.output_file + "__ADJ__", "w")
    f_r = open(opts.output_file + "__ADVERB__", "w")
    f_v = open(opts.output_file + "__VERB__", "w")

    f1 = open(opts.output_file + "__BEST__", "w")
    f1_n = open(opts.output_file + "__BEST__NOUN__", "w")
    f1_a = open(opts.output_file + "__BEST__ADJ__", "w")
    f1_r = open(opts.output_file + "__BEST__ADVERB__", "w")
    f1_v = open(opts.output_file + "__BEST__VERB__", "w")

    all_words = []
    for el in tree.findall('lexelt'):
        for ch in el.getchildren():
            for ch1 in ch.getchildren():
                sentence = ch1.text
                
                #print sentence, word, index
                word = str(el.items()[0][1])
                if int(ch.items()[0][1]) < int(opts.start_word) or int(ch.items()[0][1]) > int(opts.end_word):
                    continue 
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
                                           float(opts.thesaurus), opts.pos_context)
                #all_words.extend(result)
                values = ";".join(result[1])
                #print sentence, result
                f.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " ::: " + values)
                f1.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " :: " + values.split(";")[0])
                f.write("\n")
                f1.write("\n")
                f.flush()
                f1.flush()
                
                if word[-1] == "n":
                    f_n.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " ::: " + values)
                    f1_n.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " :: " + values.split(";")[0])
                    f_n.write("\n")
                    f1_n.write("\n")
                    f_n.flush()
                    f1_n.flush()
                if word[-1] == "j":
                    f_a.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " ::: " + values)
                    f1_a.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " :: " + values.split(";")[0])
                    f_a.write("\n")
                    f1_a.write("\n")
                    f_a.flush()
                    f1_a.flush()
                if word[-1] == "r":
                    f_r.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " ::: " + values)
                    f1_r.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " :: " + values.split(";")[0])
                    f_r.write("\n")
                    f1_r.write("\n")
                    f_r.flush()
                    f1_r.flush()
                if word[-1] == "v":
                    f_v.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " ::: " + values)
                    f1_v.write(str(el.items()[0][1]) + " " + str(ch.items()[0][1]) + " :: " + values.split(";")[0])
                    f_v.write("\n")
                    f1_v.write("\n")
                    f_v.flush()
                    f1_v.flush()

    #fx = open('loveujaaneman', 'w')
    #fx.write(str(all_words))
    #fx.close()

    f.close()
    f1.close()
    f_n.close()
    f1_n.close()
    f_a.close()
    f1_a.close()
    f_v.close()
    f1_v.close()
    f_r.close()
    f1_r.close()
    print "Output file written."
