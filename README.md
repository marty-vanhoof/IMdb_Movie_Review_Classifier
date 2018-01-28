# IMDb Movie Review Classifier

This project is my first attempt at using Keras to implement a neural network model.  My approach is to use grid search for parameter tuning and to tune a few parameters at a time and then aggregate the results, rather than using a giant parameter grid to tune them all at once.  There are a few reasons for doing this:  1) Doing a big grid search would consume too much memory on my little laptop; 2) Tuning the hyper-parameters this way the first time is better for my learning process. 
 
The project requires Python 3.x and the following Python libraries:

- [Numpy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Keras](https://keras.io/) with [TensorFlow backend](https://keras.io/backend/)
- [scikit-learn](http://scikit-learn.org/stable/)

The report for the project is written in a Jupyter Notebook and is available [here](https://github.com/marty-vanhoof/IMdb_Movie_Review_Classifier/blob/master/classifying_reviews_with_keras.ipynb).  

The Python script for training and getting the model results is [here](https://github.com/marty-vanhoof/IMdb_Movie_Review_Classifier/blob/master/imdb_models.py).  It's not a good idea to try to train neural network models in a Jupyter Notebook as the process tends to be very slow and will often just crash.  In fact, it's not a great idea to try to train neural networks that are too big on a laptop CPU either.  I'm going to eventually look into some form of cloud computing option, such as [Amazon Web Services](https://aws.amazon.com/).