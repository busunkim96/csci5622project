Contributor: Eirian Perkins

BallTree using minkowski metric, p = 5. minkowski seemed to perform the best.

eiriano> python knn-CASIA.py --kfold  --MNIST --limit 3500
    Using MNIST data
    % right:  0.944680851064
    % right:  0.931623931624
    % right:  0.921203438395
    % right:  0.93553008596
    % right:  0.941176470588
overall %:  0.934842955526

eiriano> python knn-CASIA.py --kfold  --limit 3500
    Using CASIA data
    % right:  0.910638297872
    % right:  0.901569186876
    % right:  0.927142857143
    % right:  0.925501432665
    % right:  0.915229885057
overall %:  0.916016331923

Here's the validation set (MNIST):

eiriano> python knn-CASIA.py --limit 3500 --MNIST
    Data limit: 3500
    Done loading data
        0    1    2    3    4    5    6    7    8    9
        ------------------------------------------------------------------------------
        0:    972    1    3    1    1    2    8    2    0    1
        1:    0    1053    5    0    1    0    1    3    1    0
        2:    7    12    909    11    6    2    8    26    6    3
        3:    1    1    7    954    3    30    2    5    22    5
        4:    0    12    0    0    909    0    3    8    0    51
        5:    5    2    1    27    4    831    25    2    6    12
        6:    10    1    0    0    1    4    951    0    0    0
        7:    0    14    2    1    7    2    0    1036    0    28
        8:    4    19    4    28    5    24    7    13    881    24
        9:    3    3    2    11    19    8    1    26    1    887
        Accuracy: 0.938300
