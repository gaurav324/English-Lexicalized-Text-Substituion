# This reads in the dictionary downloaded from the BigHugeThesaurus site.
class BigThesaurus:
    def __init__(self):
        _location_ = "./thesaurus_data"
        f = open(_location_, "r")
        self._data_ = eval(f.read())
        f.close()

    def word_pos_split(self, word):
        pos = word[-1]
        word = word[:-2]
        if pos == 'n':
            return (word, "noun")
        elif pos == 'v':
            return (word, "verb")
        elif pos == 'j':
            return (word, "adjective")
        elif pos == 'r':
            return (word, "adverb")

    def synonyms(self, word):
        word, pos = self.word_pos_split(word)

        syns = eval(self._data_[word][0])[pos]
        return syns["syn"] if "syn" in syns else []

    def antonyms(self, word):
        word, pos = self.word_pos_split(word)

        ants = eval(self._data_[word][0])[pos]
        return ants["ant"] if "ant" in ants else None

    def replacements(self, word):
        word, pos = self.word_pos_split(word)

        arr = eval(self._data_[word][0])[pos]
        syns = arr["syn"] if "syn" in arr else [] 
        sims = arr["sim"] if "sim" in arr else [] 

        return syns + sims
       
 

       
