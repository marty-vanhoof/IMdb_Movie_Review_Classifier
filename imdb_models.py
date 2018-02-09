import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adagrad
from sklearn.model_selection import GridSearchCV 
from keras.constraints import max_norm

def get_imdb_data(top_words=5000, max_review_length=500):

    # Load the data and split it into 50% train, 50% test sets
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

    # sequence padding
    X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

    print('number of training examples: ', len(X_train))
    print('number of testing examples: ', len(X_test))
    print('train/test set dimensions: ', X_train.shape, X_test.shape, '\n')

    return X_train, y_train, X_test, y_test

#get_imdb_data()

def build_model(optimizer='adagrad', learn_rate=0.01, learn_rate_decay=0.01, activation='relu',
                dropout_rate=0.1, weight_constraint=4, neurons=250, input_dim=5000, output_dim=32,
                max_review_length=500):
    
    model = Sequential()

    # embedding layer
    model.add(Embedding(input_dim, output_dim, input_length=max_review_length))
    model.add(Flatten())
    
    # first hidden layer 
    model.add(Dense(neurons, activation=activation, kernel_constraint=max_norm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    
    # second hidden layer
    model.add(Dense(neurons, activation=activation, kernel_constraint=max_norm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    
    # output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    optimizer = Adagrad(lr=learn_rate, decay=learn_rate_decay)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def grid_search(model, param_grid):

    # get training data
    X_train, y_train = get_imdb_data()[0], get_imdb_data()[1]

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

    # wrapper for sklearn API
    model = KerasClassifier(build_fn=build_model, verbose=1)

    # hyper-parameters for grid search
    batch_size = [100, 200, 500]
    epochs = [2, 3, 4]
    optimizer = ['adam', 'rmsprop', 'sgd', 'adagrad', 'adadelta', 'adamax', 'nadam']
    learn_rate = [0.001, 0.01, 0.1, 0.3]
    learn_rate_decay = [0.0, 0.1, 0.01, 0.001]
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
    dropout_rate = [0.1, 0.2, 0.3, 0.4]
    weight_constraint = [1,2,3,4] 
    neurons = [50, 150, 250]

    # change param_grid to grid search different hyper-parameters
    param_grid = dict(batch_size=batch_size, epochs=epochs, neurons=neurons) 

    # get the results from grid_search()
    grid_result, final_results = grid_search(model, param_grid)

    # grid_result.best_estimator_.model returns the (unwrapped) keras model
    best_model = grid_result.best_estimator_.model

    # save classifier to a .hdf5 file
    filename = 'imdb_results/imdb_neurons_best.hdf5'
    best_model.save_weights(filename)

    # write final_results dict to a csv file
    df = pd.DataFrame.from_dict(final_results)
    df.to_csv('imdb_results/grid_neurons_results.csv', index=False)

    print(df)

if __name__ == "__main__":
    main()
