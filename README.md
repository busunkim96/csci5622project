# csci5622project
Project for CSCI 5622 at CU Boulder.
Alex Okeson
Bu Sun Kim
Eirian Perkins

How to open casia.pkl.gz:
```python
>>> import cPickle
>>> import gzip
>>> f = gzip.open("casia.pkl.gz", 'rb')
>>> train_set, test_set = cPickle.load(f)
>>> x_train, y_train = train_set
>>> x_test, y_test = test_set
>>> len(x_test)
917
>>> len(x_train)
3664
>>> print y_train[400]
七
>>> print y_test[400]
二
>>>
```

the data was loaded in like:
```python
>>> for sample in samples:
...     with open ("./test/tmp/" + sample) as f:
...             img_data = f.read()
...     decoded_img_data = skimage.io.imread(StringIO(img_data))
...     x_test.append(decoded_img_data)
...     if "零" in sample:
...             y_test.append("零")
...     elif "一" in sample:
...             y_test.append("一")
...     elif "二" in sample:
...             y_test.append("二")
...     elif "三" in sample:
...             y_test.append("三")
...     elif "四" in sample:
...             y_test.append("四")
...     elif "五" in sample:
...             y_test.append("五")
...     elif "六" in sample:
...             y_test.append("六")
...     elif "七" in sample:
...             y_test.append("七")
...     elif "八" in sample:
...             y_test.append("八")
...     elif "九" in sample:
...             y_test.append("九")
...     elif "十" in sample:
...             y_test.append("十")
```
