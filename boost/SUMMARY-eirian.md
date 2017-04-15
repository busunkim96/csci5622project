# note this is actually validation set (for MNIST only, don't run this for CASIA because there is no validation set)
# so, safely ignore the "USING TEST DATA" warning
eiriano> python boost-eirian.py --MNIST
Using MNIST data
!!! USING TEST DATA !!!
Done loading data
     0	     1	2	3	4	5	6	7	8	9
------------------------------------------------------------------------------------------
0:	0	869	51	0	0	31	27	4	9	0
1:	0	0	2	0	0	48	10	18	986	0
2:	0	24	7	0	0	242	633	23	61	0
3:	0	28	76	0	0	48	28	730	120	0
4:	0	4	9	0	0	118	692	12	148	0
5:	0	39	559	0	0	70	29	119	99	0
6:	0	25	9	0	0	811	77	2	43	0
7:	0	11	9	0	0	869	36	8	157	0
8:	0	8	43	0	0	30	41	60	827	0
9:	0	8	10	0	0	94	97	19	733	0
Accuracy: 0.098900

eiriano> python boost-eirian.py --kfold  --limit 3000
Using CASIA data
      % right:  0.339149400218
      % right:  0.284623773173
      % right:  0.306434023991
      % right:  0.376226826609
      % right:  0.359869138495
overall %:  0.333260632497
eiriano> fg
vim boost-eirian.py


eiriano> python boost-eirian.py --kfold  --limit 3000 --MNIST
Using MNIST data
      % right:  0.4621
      % right:  0.5124
      % right:  0.5624
      % right:  0.4437
      % right:  0.5325
overall %:  0.50262
