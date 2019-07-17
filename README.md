# PTCinhibitors
The code contains two python files: main.py and reression.py.

main.py contains all the methods for extracting the features from a *.mol2 file (which provides the predictor features) and from a *.sort file, which contains the corresponding binding energy for every molecule.

The main function is getDataMatrix(f1,f2 -optinal) which gets as input two files: f1 - the *.mol2 file and f2 - the *.sort file. 
The function returns a Panda data frame with the following columns 'NAME', 'LOC1', 'LOC1DIST', 'LOC1STD', ... , 'LOC11', 'LOC11DIST', 'LOC11STD','BOND'
The computation of the the features is explained in the supplementary to the paper.


the second file, regression.py contains examples of training a random forrest + adaboost models, and them testing them on the various data sets.
