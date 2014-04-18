from composes.semantic_space.space import Space
from composes.transformation.scaling.ppmi_weighting import PpmiWeighting
from composes.transformation.dim_reduction.svd import Svd
from composes.transformation.feature_selection.top_feature_selection import TopFeatureSelection
from composes.similarity.cos import CosSimilarity
from composes.utils import io_utils
from composes.utils import log_utils

import sys
if __name__ == '__main__':
    # set constants
    data_path = sys.argv[0] + "/" + sys.argv[1] + "_"

    log_file = data_path + "all.log"
    core_cooccurrence_file = data_path + "GemmaData_sm"
    core_row_file = data_path + "GemmaData_rows"
    core_col_file = data_path + "GemmaData_cols"
    core_space_file = data_path + "core.pkl"
    
    # config log file
    log_utils.config_logging(log_file)
    
    print "Building semantic space from co-occurrence counts"
    core_space = Space.build(data=core_cooccurrence_file, rows=core_row_file,
                             cols=core_col_file, format="sm")
    
    print "Applying ppmi weighting"
    core_space = core_space.apply(PpmiWeighting())
    print "Applying feature selection"
    core_space = core_space.apply(TopFeatureSelection(5000))
    print "Applying svd 500"
    core_space = core_space.apply(Svd(100))
    
    print "Saving the semantic space"
    io_utils.save(core_space, core_space_file)
    
    #print "Finding 10 neighbors of " + sys.argv[1]
    #neighbors = core_space.get_neighbours(sys.argv[1], 10, CosSimilarity())
    #print neighbors
