# About
Python package for supervised &amp; unsupervised learning.

Algorithms for **supervised learning** include:
- naive Bayes classifier (`bayes`);
- deep neural networks (`deep`);
- kernel perceptron algorithm (`kerpercep`);
- kernel support vector machines (`kersvm`);
- *k*-nearest neighbors (`knns`);
- perceptron algorithm (`percep`);
- shallow neural networks (`shallow`);
- support vector machines (`svm`).

Algorithms for **unsupervised learning** include:
- *k*-means (`kmeans`).

Algorithms for **time series** include:
- autoregression of order *p* (`arp`);
- autocorrelation function (`acf`);
- partial autocorrelation computation (`pacf`).

Also included:
- Monte Carlo integration (`mtcarlo`);
- princpal componenent analysis (`pca`);
- regression (`reg`).

The **examples** folder contains examples for each of these functions. 

The **memos** folder contains PDF files about:
- probability (`proba1.pdf` &amp; `proba2.pdf`); 
- statistics (`stats1.pdf`, `stats2.pdf` &amp; `stats3.pdf`);
- supervised learning (`supervised.pdf`).

# Installation

To install, you can either download a .zip file or clone the directory with Git.

### Option 1: Download a .zip file

Download a .zip of learnpy from:

- https://github.com/Hadrien-Montanelli/learnpy/archive/master.zip

### Option 2: Clone with Git

To clone the learnpy repository, first navigate in a terminal to where you want the repository cloned, then type:
```
git clone https://github.com/Hadrien-Montanelli/learnpy.git
```
### PYTHONPATH
Once you have a copy of learnpy, you have to add it to your PYTHONPATH. If you're using an editor like Spyder, you can do this with the integrated PYTHONPATH manager. 

Otherwise, you can directly change your PYTHONPATH. For example, on a Mac, edit your `~/.bash_profile` and add:
```
export PYTHONPATH="${PYTHONPATH}:/path_to_learnpy/"
```
Don't forget to `souce ~/.bash_profile` when you're done.

# Getting started 

I recommend taking a look at the **examples** folder. To get a taste of what computing with learnpy is like, type:
```python
import numpy as np
from learnpy.timeseries import arp, acf, pacf
        
x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)

arp(x_t, 2)           # auto-regression with parameter p=2
acf(x_t)              # compute the autocorrelation function
pacf(x_t)             # compute the partial autocorrelation function
```

The next example trains a shallow network with 15 neurons; the dataset has 20 features, the training set contains 4,000 data points while the testing set has 1,000 points:
```python
from sklearn import datasets
from learnpy.supervised import shallow

X, Y = datasets.make_classification(5000, random_state = 123)

X_train, X_test = X[:4000], X[4000:]        # data points (training set & testing set)
Y_train, Y_test = Y[:4000], Y[4000:]        # labels (training set & testing set)
classifier = shallow(20, 4000, 15)          # shallow network (20 features, 4,000 points, 15 neurons)
classifier.train(X_train, Y_train)          # train
Y_hat = classifier.classify(X_test)         # classify
acc = classifier.accuracy(Y_test, Y_hat)    # compute the accuracy
print(f'Accuracy: {acc}%')
```
We've just trained and tested a shallow network that has a single hidden layer (15 neurons with ReLU activation functions) and an output layer (sigmoid).

Deep neural networks are also supported:
```python
import numpy as np
from sklearn import datasets
from learnpy.supervised import deep

X, Y = datasets.make_classification(5000, random_state = 123)

X_train, X_test = X[:4000], X[4000:]                # data points (training set & testing set)
Y_train, Y_test = Y[:4000], Y[4000:]                # labels (training set & testing set)
classifier = deep(20, 4000, 2, np.array([10, 12]))  # deep network (2 hidden layers with 10 & 12 neurons)
classifier.train(X_train, Y_train)                  # train
Y_hat = classifier.classify(X_test)                 # classify
acc = classifier.accuracy(Y_test, Y_hat)            # compute the accuracy
print(f'Accuracy: {acc}%')
```

Any comment or question, send an email to: hadrien.montanelli@gmail.com

# License
See `LICENSE.txt` for licensing information.
