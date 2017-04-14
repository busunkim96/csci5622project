BallTree using minkowski metric, p = 5. minkowski seemed to perform the best.

eiriano> python knn-CASIA.py --kfold True --MNIST --limit 3500
Using MNIST data
% right:  0.931434282859
% right:  0.929414117177
% right:  0.9334
% right:  0.933480044013
% right:  0.935174069628
overall %:  0.932580502735

eiriano> python knn-CASIA.py --kfold True  --limit 3500
Using CASIA data
% right:  0.960704607046
% right:  0.952316076294
% right:  0.954979536153
% right:  0.954794520548
% right:  0.942386831276
overall %:  0.953036314263
