import time
import logging
from numpy import array
from numpy import prod
from composes.utils.space_utils import list2dict
from composes.utils.space_utils import assert_dict_match_list
from composes.utils.space_utils import assert_shape_consistent
from composes.utils.gen_utils import assert_is_instance
from composes.utils.space_utils import add_items_to_dict
from composes.utils.matrix_utils import resolve_type_conflict
from composes.utils.matrix_utils import get_type_of_largest
from composes.matrix.matrix import Matrix
from composes.matrix.dense_matrix import DenseMatrix
from composes.matrix.sparse_matrix import SparseMatrix
from composes.semantic_space.operation import FeatureSelectionOperation
from composes.semantic_space.operation import DimensionalityReductionOperation
from composes.similarity.similarity import Similarity
from composes.transformation.scaling.scaling import Scaling
from composes.transformation.dim_reduction.dimensionality_reduction import DimensionalityReduction
from composes.transformation.feature_selection.feature_selection import FeatureSelection
from composes.exception.illegal_state_error import IllegalOperationError
from composes.utils import log_utils as log
from composes.utils.io_utils import read_sparse_space_data
from composes.utils.io_utils import extract_indexing_structs
from composes.utils.io_utils import read_dense_space_data
from composes.utils.io_utils import create_parent_directories
from composes.utils.io_utils import print_list
from composes.utils.io_utils import print_cooc_mat_dense_format
from composes.utils.io_utils import print_cooc_mat_sparse_format

logger = logging.getLogger(__name__)

from composes.semantic_space.space import Space
class MySpace(Space):

    @classmethod
    def xbuild(cls, **kwargs):
        """
        Reads in data files and extracts the data to construct a semantic space.

        If the data is read in dense format and no columns are provided,
        the column indexing structures are set to empty.

        Args:
            data: file containing the counts
            format: format on the input data file: one of sm/dm
            rows: file containing the row elements. Optional, if not provided,
                extracted from the data file.
            cols: file containing the column elements

        Returns:
            A semantic space build from the input data files.

        Raises:
            ValueError: if one of data/format arguments is missing.
                        if cols is missing and format is "sm"
                        if the input columns provided are not consistent with
                        the shape of the matrix (for "dm" format)

        """
        start = time.time()
        id2row = None
        id2column = None

        if "data" in kwargs:
            data_file = kwargs["data"]
        else:
            raise ValueError("Space data file needs to be specified")

        if "format" in kwargs:
            format_ = kwargs["format"]
            if not format_ in ["dm","sm"]:
                raise ValueError("Unrecognized format: %s" % format_)
        else:
            raise ValueError("Format of input files needs to be specified")

        if "rows" in kwargs and not kwargs["rows"] is None:
            [id2row], [row2id] = extract_indexing_structs(kwargs["rows"], [0])

        if "cols" in kwargs and not kwargs["cols"] is None:
            [id2column], [column2id] = extract_indexing_structs(kwargs["cols"], [0])
        elif format_ == "sm":
            raise ValueError("Need to specify column file when input format is sm!")

        if format_ == "sm":
            if id2row is None:
                [id2row], [row2id] = extract_indexing_structs(data_file, [0])
            mat = read_sparse_space_data(data_file, row2id, column2id)

        else:
            if id2row is None:
                [id2row],[row2id] = extract_indexing_structs(data_file, [0])
            mat = read_dense_space_data(data_file, row2id)

        if id2column and len(id2column) != mat.shape[1]:
            raise ValueError("Columns provided inconsistent with shape of input matrix!")

        if id2column is None:
            id2column, column2id = [], {}

        log.print_matrix_info(logger, mat, 1, "Built semantic space:")
        log.print_time_info(logger, time.time(), start, 2)
        return MySpace(mat, id2row, id2column, row2id, column2id)

    def get_xneighbours(self, vector, no_neighbours, similarity,
                       space2=None):
        """
        Computes the neighbours of a word in the semantic space.

        Args:
            word: string, target word
            no_neighbours: int, the number of neighbours desired
            similarity: of type Similarity, the similarity measure to be used
            space2: Space type, Optional. If provided, the neighbours are
                retrieved from this space, rather than the current space.
                Default, neighbours are retrieved from the current space.

        Returns:
            list of (neighbour_string, similarity_value) tuples.

        Raises:
            KeyError: if the word is not found in the semantic space.

        """

        start = time.time()
        assert_is_instance(similarity, Similarity)

        if space2 is None:
            id2row = self.id2row
            sims_to_matrix = similarity.get_sims_to_matrix(vector,
                                                          self.cooccurrence_matrix)
        else:
            mat_type = type(space2.cooccurrence_matrix)
            if not isinstance(vector, mat_type):
                vector = mat_type(vector)

            sims_to_matrix = similarity.get_sims_to_matrix(vector,
                                         space2.cooccurrence_matrix)
            id2row = space2.id2row

        sorted_perm = sims_to_matrix.sorted_permutation(sims_to_matrix.sum, 1)
        no_neighbours = min(no_neighbours, len(id2row))
        result = []

        for count in range(no_neighbours):
            i = sorted_perm[count]
            result.append((id2row[i], sims_to_matrix[i,0]))

        log.print_name(logger, similarity, 1, "Similarity:")
        log.print_time_info(logger, time.time(), start, 2)
        return result

    def apply(self, transformation):
        """
        Applies a transformation on the current space.

        All transformations affect the data matrix. If the transformation
        reduces the dimensionality of the space, the column indexing
        structures are also updated. The operation applied is appended
        to the list of operations that the space holds.

        Args:
            transformation: of type Scaling, DimensionalityReduction or
              FeatureSelection

        Returns:
            A new space on which the transformation has been applied.

        """
        start = time.time()
        #TODO , FeatureSelection, DimReduction ..
        assert_is_instance(transformation, (Scaling, DimensionalityReduction,
                                            FeatureSelection))
        op = transformation.create_operation()
        new_matrix =  op.apply(self.cooccurrence_matrix)

        new_operations = list(self.operations)
        new_operations.append(op)

        id2row, row2id = list(self.id2row), self.row2id.copy()


        if isinstance(op, DimensionalityReductionOperation):
            self.assert_1dim_element()
            id2column, column2id = [], {}
        elif isinstance(op, FeatureSelectionOperation):
            self.assert_1dim_element()
            op.original_columns = self.id2column

            if op.original_columns:
                id2column = list(array(op.original_columns)[op.selected_columns])
                column2id = list2dict(id2column)
            else:
                id2column, column2id = [],{}
        else:
            id2column, column2id = list(self.id2column), self.column2id.copy()

        log.print_transformation_info(logger, transformation, 1,
                                      "\nApplied transformation:")
        log.print_matrix_info(logger, self.cooccurrence_matrix, 2,
                              "Original semantic space:")
        log.print_matrix_info(logger, new_matrix, 2, "Resulted semantic space:")
        log.print_time_info(logger, time.time(), start, 2)

        return MySpace(new_matrix, id2row, id2column,
                     row2id, column2id, operations = new_operations)
