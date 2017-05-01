# note this is actually validation set (for MNIST only, don't run this for CASIA because there is no validation set)
# so, safely ignore the "USING TEST DATA" warning
eiriano> python boost-eirian.py --MNIST --limit 3500
Using MNIST data
!!! USING TEST DATA !!!
Data limit: 3500
Done loading data
Accuracy: 0.911400

eiriano> python boost-eirian.py --MNIST --limit 3500 --kfold
Using MNIST data
% right:  0.9075
% right:  0.9114
% right:  0.9094
% right:  0.9044
% right:  0.9056
overall %:  0.90766

eiriano> python boost-eirian.py  --limit 3500 --kfold
Using CASIA data
% right:  0.878953107961
% right:  0.896401308615
% right:  0.900763358779
% right:  0.906215921483
% right:  0.884405670665
overall %:  0.893347873501


#######
With preprocessing

eiriano> python boost-eirian.py --limit 3500 --kfold
Using CASIA data
% right:  0.932388222465
% right:  0.935659760087
% right:  0.924754634678
% right:  0.922573609597
% right:  0.934569247546
overall %:  0.929989094875

eiriano> fg
python boost-eirian.py --limit 3500 --kfold --MNIST
% right:  0.909
% right:  0.909
% right:  0.9107
% right:  0.9114
% right:  0.9092
overall %:  0.90986
