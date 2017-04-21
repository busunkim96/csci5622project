# note this is actually validation set (for MNIST only, don't run this for CASIA because there is no validation set)
# so, safely ignore the "USING TEST DATA" warning
```
eiriano> python boost-eirian.py --MNIST --limit 3500
Using MNIST data
!!! USING TEST DATA !!!
Data limit: 3500
Done loading data
Accuracy: 0.911400
```
```
eiriano> python boost-eirian.py --MNIST --limit 3500 --kfold
Using MNIST data
% right:  0.9075
% right:  0.9114
% right:  0.9094
% right:  0.9044
% right:  0.9056
overall %:  0.90766
```
```
eiriano> python boost-eirian.py  --limit 3500 --kfold
Using CASIA data
% right:  0.878953107961
% right:  0.896401308615
% right:  0.900763358779
% right:  0.906215921483
% right:  0.884405670665
overall %:  0.893347873501
```
