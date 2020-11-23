# About
Python package for supervised &amp; unsupervised learning, and times series.

Algorithms for **supervised learning** include:
- naive Bayes classifier (`bayes`);
- *k*-nearest neighbors (`knns`);
- perceptron algorithm (`percep`);
- kernel perceptron algorithm (`kerpercep`);
- support vector machines (`svm`);
- kernel support vector machines (`kersvm`);
- shallow neural networks (`shallow`).

Algorithms for **unsupervised learning** include:
- *k*-means (`kmeans`).

Algorithms for **time series** include:
- autoregression (`ar`);
- autocorrelation computation (`autocorr`);
- partial autocorrelation computation (`pautocorr`).

Also included:
- Monte Carlo integration (`montecarlo`);
- princpal componenent analysis (`pca`);
- regression (`regression`).

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
from learnpy.timeseries import ar, autocorr, pautocorr
        
x_t = np.linspace(-1, 1, 100) + 5e-1*np.random.randn(100)

ar(x_t, 2)                 # auto-regression with parameter p=2
autocorr(x_t)              # compute the autocorrelation function
pautocorr(x_t)             # compute the partial autocorrelation function
```

The following example trains a shallow network with 10 neurons:
```python
from sklearn import datasets
from learnpy.supervised import shallow

X, Y = datasets.make_classification(5000, random_state = 123)

X_train, X_test = X[:4000], X[4000:]              # data points (20 features)
Y_train, Y_test = Y[:4000], Y[4000:]              # labels
classifier = shallow(20, 4000, 10)                # create a shallow network with 10 neurons
classifier.train(X_train, Y_train)                # train
Y_hat = classifier.classify(X_test)               # classify
acc = classifier.accuracy(Y_test, Y_hat)          # compute the accuracy
print(f'Accuracy: {acc}%')
```

Any comment or question, send an email to: hadrien.montanelli@gmail.com

# License
See `LICENSE.txt` for licensing information.
