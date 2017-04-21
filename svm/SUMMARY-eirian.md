'''
eiriano> python svm-eirian.py --kfold --limit 3500
Using CASIA data
      % right:  0.929116684842
      % right:  0.940021810251
      % right:  0.930207197383
      % right:  0.925845147219
      % right:  0.930207197383
overall %:  0.931079607415
'''

'''
eiriano> python svm-eirian.py --kfold --limit 3500 --MNIST
Using MNIST data
      % right:  0.9206
      % right:  0.9226
      % right:  0.9185
      % right:  0.9211
      % right:  0.9223
overall %:  0.92102
eiriano> fg
vim svm-eirian.py
'''
'''
eiriano> # this is with validation set, not test set:
eiriano> python svm-eirian.py --limit 3500 --MNIST
Using MNIST data
!!! USING TEST DATA !!!
Data limit: 3500
Done loading data
Accuracy: 0.923100
'''

Standardization seems to make no aprpeciable difference (mean removal + variance scaling)
'''
(python2) busun@busun-Lenovo-IdeaPad-P500:~/csci5622project/svm$ python svm-eirian.py --kfold --limit 3500 --standardize 
Using CASIA data
      % right:  0.931297709924
      % right:  0.930207197383
      % right:  0.930207197383
      % right:  0.924754634678
      % right:  0.930207197383
overall %:  0.92933478735
'''

'''
(python2) busun@busun-Lenovo-IdeaPad-P500:~/csci5622project/svm$ python svm-eirian.py --kfold --limit 3500 --standardize --MNIST
Using MNIST data
      % right:  0.9191
      % right:  0.9213
      % right:  0.9226
      % right:  0.9215
      % right:  0.9221
overall %:  0.92132
'''