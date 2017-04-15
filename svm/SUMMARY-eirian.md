# haven't played with tuning parameters yet

eiriano> python svm-eirian.py --kfold
Using CASIA data
      % right:  0.88985823337
      % right:  0.892039258451
      % right:  0.894220283533
      % right:  0.88985823337
      % right:  0.894220283533
overall %:  0.892039258451

eiriano> python svm-eirian.py --kfold --limit 3500 --MNIST
Using MNIST data
      % right:  0.8994
      % right:  0.8983
      % right:  0.9003
      % right:  0.9003
      % right:  0.8984
overall %:  0.89934
