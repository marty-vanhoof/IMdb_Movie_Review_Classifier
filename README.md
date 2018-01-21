# IMDb Movie Review Classifier

This project is my first attempt at using Keras to implement a neural network model.  It requires Python 3.x and the following Python libraries:

- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Keras](https://keras.io/) with [TensorFlow backend](https://keras.io/backend/)
- [scikit-learn](http://scikit-learn.org/stable/)

The report for the project is written in a Jupyter Notebook and is available [here](https://github.com/marty-vanhoof/IMdb_Movie_Review_Classifier/blob/master/classifying_reviews_with_keras.ipynb).  The Python script for training and getting the model results is [here](https://github.com/marty-vanhoof/IMdb_Movie_Review_Classifier/blob/master/imdb_models.py).  It's not a good idea to try to train neural network models in a Jupyter Notebook as the process tends to be very slow and will often just crash.  In fact, it's not a great idea to try to train neural networks that are too big on a laptop CPU either.  I'm going to eventually look into some form of cloud computing option, such as [Amazon Web Services](https://aws.amazon.com/).