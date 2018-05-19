# IMDb Movie Review Classifier

This project is my first attempt at using Keras to implement a neural network model.  My approach is to use grid search for parameter tuning and to tune a few parameters at a time and then aggregate the results, rather than using a giant parameter grid to tune them all at once.  There are a few reasons for doing this:  1) Doing a big grid search would consume too much memory on my little laptop; 2) Tuning the hyper-parameters this way the first time is better for my learning process. 
 
The project requires Python 3.x and the following Python libraries:

- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Keras](https://keras.io/) with [TensorFlow backend](https://keras.io/backend/)
- [scikit-learn](http://scikit-learn.org/stable/)

The report for the project can be viewed in jupyter nbviewer and is available [here](https://nbviewer.jupyter.org/github/marty-vanhoof/IMdb_Movie_Review_Classifier/blob/master/classifying_reviews_with_keras.ipynb), although there are a couple places at the beginning of the report where the output is not rendered properly.  The original jupyter notebook is [here](classifying_reviews_with_keras.ipynb).  

The Python script for training and getting the model results is [here](https://github.com/marty-vanhoof/IMdb_Movie_Review_Classifier/blob/master/imdb_models.py).  If you are not using some form of cloud computing, then trying to train neural network models in a Jupyter Notebook on your local machine is frustrating, as the process tends to be very slow and will often just crash.  Running the `.py` file directly in the terminal is faster, but in general it's not a good idea to to try train neural network models that are too big on a laptop CPU.  A good option is some form of cloud computing, such as [Amazon Web Services](https://aws.amazon.com/).

I managed to get an accuracy of 88.38% with this model, which is just a regular MLP model.  Not bad for a first go at this.  The original [2011 paper](http://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf) by Maas et al. achieved an accuracy of 88.89% and was considered state-of-the-art at the time.  Of course, this is pretty old in deep learning terms, since the field is developing so quickly.  These days, people have used convolutional neural networks to get much better results.  