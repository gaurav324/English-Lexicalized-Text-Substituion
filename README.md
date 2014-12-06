English-Lexicalized-Text-Substituion
====================================

Akanksha and I did this project as a part of Natural Language Processing course and Graphical Model.
English Lexicalized substiution task (http://nlp.cs.swarthmore.edu/semeval/tasks/task10/summary.shtml) was introduced in SEMEVAL-2007. After that many people have worked on this interesting problem. 

We approached this task by exploring distributional semantics of the sentences. Our results are very impresssive. We have been
able to achieve state-of-the-art numbers for one of the evaluation metric. 

We have then also approached this problem from a graphical model perspective. WE then also compare the results of these two approaches.  

Refer to the pdf for details.

## How to test.
You need DISSECT TOOlkit installed to test this.

python ./scripts/dissect_model_tester.py --pkl_file ./data/1_lemma_pos.pkl --xml_input ./TaskTestData/test/lexsub_test_cleaned.xml --top_features 5000 --ppmi --lwindow 1 --rwindow 1  --left_right_add  --thesaurus 1.0 --output_to some_random_file

## Evaluation Metric.
perl ./TaskTestData/key/score.pl some_random_file__BEST__ ./TaskTestData/key/gold -t best
perl ./TaskTestData/key/score.pl some_random_file ./TaskTestData/key/gold -t oot

# Source code 
https://github.com/gaurav324/English-Lexicalized-Text-Substituion.git
## Directory Structure:
    - ipython_notebooks/
      This is ipython test code, that we were writing and sharing among each other
      while we were testing various features. Evaluator can ignore this.

    - data/
    We parsed data from two data source and pre-processed it to produce pickle files.
    readable by python. As the size of actual pickle files is huge, we are turning in one
    411M of decent size pkl file. You can use this file for testing.

    - TaskTestData/
    This code has been provided by the task organizers to test the performance of our model.
    Use the following commands to execute scoring script:
    > perl TaskTestData/key/score.pl <OUTPUT_FILE> ../TaskTestData/key/gold -t best
    > perl TaskTestData/key/score.pl <OUTPUT_FILE> ../TaskTestData/key/gold -t oot

    - scripts/
    a) BigThesaurus.py
        - Used by our model to get candidate substitutions.
        - Teser can Ignore.
    b) SpaceSubClassModel.py
        - We subclass Space class from the Dissect python toolkit to support
          advanced vector operations.
        - If using our model, one must have Dissect toolkit installed on their machines.
        http://clic.cimec.unitn.it/composes/toolkit/
        - Tester can ignore.
    c) combine_sm_models.py
        - This script is used to combine co-occurance count files generated from
        different data sources.
        - Tester can ignore.
    d) dissect_sample.py
        - This is a test script to run dissect model.
    e) gemma_parser.py
        - This script is used to parse WikiCorpus ttp://www.lsi.upc.edu/~nlp/wikicorpus.
    f) uwack_parser_pos.py
        - This script is used to parse huge http://wacky.sslmit.unibo.it/doku.php?id=corpora
        - Out of these parsing scirpts is co-occurance counts file, and two other files
          which Dissect expects as input.
    g) upper_bound.py
        - Test Script
        - Tester can Ignore.
    h) thesaurus_data
        - Actual thesaurus data, which we parsed from the BigHugeCorpus site.
        - Tester can Ignore.
    i) thesaurusParser.py
        - This script was used to parse thesaurus data and store it in JSON format.
        - Tester can Ignore.
    j) google_model_tester.py
        - We also ran some initial tests in google word2Vec. However, this is not used
        in our final approah.
        - Tester can Ignore.
    k) get_pos_split_on_output.py
        - Test script.
        - Tester can ignore.
    l) dissect_model_tester.py
        - This is the main script, which we run for running all kind of simulations.
        - Run python ./dissect_model_tester.py --help to get list of all options.
        - <NOTE>
            - Install Dissect tool kit before running this script.
            - You should have BigThesaurus.py and thesaurus_data in the same folder
              as this script.
        - This model can be passed in raw co-occurance counts, which can be used to 
          produce pickle files, which can be used later.
        - There are many features in this script. However, only we good combinations 
          provide us with good resutls.

       Sample invocations:

       python /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/scripts/dissect_model_tester.py --rows_file /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_row_file_pos --cols_file /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_col_file_pos --sm_file /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_matrix_file_pos --xml_input /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/TaskTestData/test/lexsub_test_cleaned.xml --top_features 5000 --save_location /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_gemma_plmi.pkl --ppmi --lwindow 1 --rwindow 1  --left_right_add  --thesaurus 1.0 --output_to gemma_plmi_lradd_1.0t_5000_test_no_pos

    --rows_file, --cols_file and --sm_file are not required, if you have pkl file. This 
    invocation would do the required computations and all also save the pickle file for 
    futture use.

    >> Equivalent invocation would be:

    python /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/scripts/dissect_model_tester.py --pkl_file /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/data/150_with_ppmi.pkl --xml_input /Users/gnanda/nlp/final_project/English-Lexicalized-Text-Substituion/TaskTestData/test/lexsub_test_cleaned.xml --top_features 5000 --ppmi --lwindow 1 --rwindow 1  --left_right_add  --thesaurus 1.0 --output_to gemma_plmi_lradd_1.0t_5000_test_no_pos

    # This would generate various output files, using thesaurus can candidate substitutions. Use 
     window size of 1 for creating context vectors. Selects top 5000 features from the vector
     space. Use PPMI (Positive Pointwise Mututal information) to select feature set.


    m) graphical_model_tester.py
        - This is the main script used to call the Graphical Model based approach used to create the final output file. 
        It can be used to run all kind of simulations.
        - Run python ./graphical_model_tester.py --help to get list of all options.
        - <NOTE>
            - Install Dissect tool kit before running this script.
            - You should have BigThesaurus.py and thesaurus_data in the same folder
              as this script.
        - This model can be passed in raw co-occurance counts, which can be used to 
          produce pickle files, which can be used later.
        - There are many features in this script, which can be used to control the tarining data, the count og how many words on the left or the right need to be included n the graph being created.
        - In order to control 
       Sample invocations:
       python scripts/graphical_model_tester.py --pkl_file ./data/1_lemma_pos.pkl --xml_input ./TaskTestData/test/lexsub_test_cleaned.xml --lwindow 1 --rwindow 1  --left_right_add  --thesaurus 1.0 --output_to output_nanda_6 --start_word 1 --end_word 2100
