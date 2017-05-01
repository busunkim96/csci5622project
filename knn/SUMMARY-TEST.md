Without preprocessing:

ing CASIA data
!!! USING TEST DATA !!!
Done loading data
    0   1   2   3   4   5   6   7   8   9   10
    ------------------------------------------------------------------------------------------
    0:  74  0   0   0   2   0   1   0   1   1   4
    1:  0   84  0   0   0   0   0   0   0   0   0
    2:  0   0   81  0   0   1   0   0   0   0   0
    3:  0   0   9   73  0   1   0   1   0   0   0
    4:  1   0   0   2   79  0   0   0   1   0   0
    5:  0   0   0   13  0   67  0   3   0   0   1
    6:  2   0   1   1   0   0   72  1   3   0   3
    7:  1   1   0   1   0   0   2   72  0   1   4
    8:  0   0   0   0   0   0   1   0   82  0   0
    9:  3   1   0   0   0   2   3   7   1   65  1
    10: 0   0   0   0   0   0   1   0   0   0   85
    Accuracy: 0.909487


Oh right, I think preprocessing actually made KNN worse 
I think these results for CASIA (above) indicate some over fitting



eiriano> python knn-CASIA.py --limit 4500 --MNIST
  /Users/eiriano/anaconda2/lib/python2.7/site-packages/sklearn/grid_search.py:43: DeprecationWarning: This module was deprecated in version 0.18 in favor of the model_selection module into which all the refactored classes and functions are moved. This module will be removed in 0.20.
    DeprecationWarning)
    Using MNIST data
    !!! USING TEST DATA !!!
    Data limit: 4500
    Done loading data
    0   1   2   3   4   5   6   7   8   9
    ------------------------------------------------------------------------------------------
    0:  964 1   3   0   0   3   7   2   0   0
    1:  0   1125    3   2   0   0   3   2   0   0
    2:  14  12  958 8   3   3   7   19  8   0
    3:  0   1   6   928 1   35  4   10  19  6
    4:  1   11  1   0   912 0   8   1   5   43
    5:  4   4   2   20  3   823 16  3   9   8
    6:  8   4   0   0   5   5   935 0   0   1
    7:  1   29  7   0   7   1   0   951 2   30
    8:  12  3   6   13  9   19  8   10  876 18
    9:  2   5   2   8   22  6   2   18  10  934
    Accuracy: 0.940600
