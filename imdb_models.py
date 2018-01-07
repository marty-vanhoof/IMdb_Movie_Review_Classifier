import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import RMSprop
from sklearn.model_selection import GridSearchCV 

def get_imdb_data(top_words=5000, max_review_length=500):

    # Load the data and split it into 50% train, 50% test sets
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # sequence padding
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    return X_train, y_train, X_test, y_test

def print_info():

    print('number of training examples: ', len(X_train))
    print('number of testing examples: ', len(X_test))

    print('train/test set dimensions: ', X_train.shape, X_test.shape, '\n')
    print('first sentence in training set, encoded and padded:\n\n ', X_train[0], '\n')

def build_model(optimizer='rmsprop', input_dim=5000, output_dim=32, input_length=500):
    
    model = Sequential()
    
    # input layer
    model.add(Embedding(input_dim, output_dim, input_length=max_review_length))
    
    # first hidden layer 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    
    # second hidden layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    
    # output layer
    model.add(Dense(1, activation='logistic'))
    
    # compile the model
    #optimizer = RMSprop(lr=learn_rate, rho=grad_decay)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def grid_search(model, param_grid):

    # set up and perform the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    # the cv_results_ attribute is a dict that summarizes many important results
    # from each model
    test_score_means = grid_result.cv_results_['mean_test_score']
    train_score_means = grid_result.cv_results_['mean_train_score']
    train_times = grid_result.cv_results_['mean_fit_time']
    params = grid_result.cv_results_['params']

    # store these results in a smaller dict for easy loading into a pandas dataframe
    final_results = dict(mean_test_score=test_score_means, mean_train_score=train_score_means,
                         mean_fit_time=train_times, params=params)

    print('Best score {} using hyperparameters {}'.format(grid_result.best_score_, grid_result.best_params_))
    
    return grid_result, final_results


def main():

    np.random.seed(13)

    # get training data
    X_train, y_train = get_imdb_data[0], get_imdb_data[1]

    # wrapper for sklearn API
    model = KerasClassifier(build_fn=build_model, batch_size=500, verbose=1)

    # hyper-parameters for grid search
    batch_size = [50, 100, 500, 1000]
    epochs = [20]
    optimizer = ['rmsprop', 'adam', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
    learn_rate = [0.0001, 0.001, 0.01, 0.1, 0.3]
    # grad_decay is called rho in the documentation (it's not learn rate decay)
    grad_decay = [0.1, 0.5, 0.9]

    # change param_grid to grid search different parameters
    param_grid = dict(epochs=epochs, learn_rate=learn_rate, grad_decay=grad_decay) 

    # get the results from grid_search()
    grid_result, final_results = grid_search(model, param_grid)

    # grid_result.best_estimator_.model returns the (unwrapped) keras model
    best_model = grid_result.best_estimator_.model

    # save classifier to a .hdf5 file
    filename = 'mnist_rms-lr-rho_best.hdf5'
    best_model.save_weights(filename)

    # write final_results dict to a csv file
    df = pd.DataFrame.from_dict(final_results)
    df.to_csv('grid_rms_results.csv', index=False)

    print(df)

if __name__ == "__main__":
    main()
