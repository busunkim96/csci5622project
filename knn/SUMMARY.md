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

