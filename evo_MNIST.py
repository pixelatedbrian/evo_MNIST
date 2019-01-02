from __future__ import print_function
import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # change to your device

import time
import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

from keras.activations import elu, relu, selu, sigmoid, hard_sigmoid, linear
from keras.layers.advanced_activations import LeakyReLU, PReLU

from keras.layers import Dense, Lambda, Input, concatenate, Layer, Dropout
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop, Adadelta

from keras import backend as K

#################################################
### Default model parameters before starting: ###
#################################################

log_file = "genome_003.h5"

PERMUT_OFF = 5  # how many modified offspring per generation (1 is always the starting value)
# input image dimensions
img_rows, img_cols = 28, 28
    
m_params = {}

# true constants that won't change during training
m_params["MAX_GOAL_TIME"] = 45.0 # target max seconds per run, penalize for more time, slightly bonus for less time
m_params["BEST_TIME_BONUS"] = 1.1 # try to prevent the alg to ONLY make things faster

# constant only for a run
m_params["BATCH_SIZE"] = 2048      # [1 - 4096] (powers of 2)
m_params["EPOCHS"] = 40           # [10 - 50]
m_params["LEARNING_RATE"] = 0.01    # [1e1 - 1e-5]
m_params["LR_DECAY"] = 1e-5        # [1e-2 - 1e-5]

m_params["OPTIMIZER"] = Adam   # [Adadelta, Adam, RMSprop, SGD]

# dropout
# applies in the gab preceeding the label
# For example DO_B occurs in between NODES_DENSE_A and NODES_DENSE_B
m_params["DO_A"] = 0.2             # [0 - 1.0]
m_params["DO_B"] = 0.2             # [0 - 1.0]
m_params["DO_C"] = 0.2             # [0 - 1.0]
m_params["DO_D"] = 0.2             # [0 - 1.0]
m_params["DO_E"] = 0.2             # [0 - 1.0]
m_params["DO_F"] = 0.2  # if there is layer E there should be ability to have DO between last FC and activation

m_params["NODES_FC_A"] = 20        # [2 - 1000]
m_params["NODES_FC_B"] = 20        # [2 - 1000]
m_params["NODES_FC_C"] = 20        # [2 - 1000]
m_params["NODES_FC_D"] = 20        # [2 - 1000]
m_params["NODES_FC_E"] = 20        # [2 - 1000]

m_params["ACT_FC_A"] = "relu"     # [elu, relu, selu, sigmoid, hard_sigmoid, linear]
m_params["ACT_FC_B"] = "relu"     # [elu, relu, selu, sigmoid, hard_sigmoid, linear]
m_params["ACT_FC_C"] = "relu"     # [elu, relu, selu, sigmoid, hard_sigmoid, linear]
m_params["ACT_FC_D"] = "relu"     # [elu, relu, selu, sigmoid, hard_sigmoid, linear]
m_params["ACT_FC_E"] = "relu"     # [elu, relu, selu, sigmoid, hard_sigmoid, linear]


def darwin(params, generations=50):
    log = []

    # in the future decide which 'gene' to MUTATE before the for loop.
    mutation_keys = ["EPOCHS", "NEURONS", "NEURON_ACTIVATION", "LEARNING_RATE", "DROPOUT"]
    # mutation_keys = ["NEURON_ACTIVATION"]

    for generation in range(generations):
        mutation_type = np.random.choice(mutation_keys)

        print("Generation {:}: {:}".format(generation + 1, mutation_type))

        ############################
        ### if mutating epochs:  ###
        ############################
        if mutation_type == "EPOCHS":
            # get current epoch
            mutations = [params[mutation_type]]

            # now make PERMUT_OFF permutations:
            while len(mutations) < PERMUT_OFF:
                temp_e = np.random.randint(-10, 10)
                temp_e += mutations[0]

                if 2 <= temp_e <= 150:
                    if temp_e not in mutations:
                        mutations.append(temp_e)

        elif mutation_type == "NEURONS":
            mutation_type = np.random.choice(["NODES_FC_A", "NODES_FC_B", "NODES_FC_C", "NODES_FC_D", "NODES_FC_E"])

            mutations = [params[mutation_type]]

            while len(mutations) < PERMUT_OFF:
                temp = np.random.randint(-25, 25)
                temp += mutations[0]  # difference off the original value

                if 2 <= temp <= 200:
                    if temp not in mutations:
                        mutations.append(temp)

        elif mutation_type == "NEURON_ACTIVATION":
            mutation_type = np.random.choice(["ACT_FC_A", "ACT_FC_B", "ACT_FC_C", "ACT_FC_D", "ACT_FC_E"])

            mutations = [params[mutation_type]]

            while len(mutations) < PERMUT_OFF:
                temp = np.random.choice(["elu", "relu", "selu", "sigmoid", "hard_sigmoid", "linear", "tanh"])

                if temp not in mutations:
                    mutations.append(temp)

        elif mutation_type == "LEARNING_RATE":
            mutations = [params[mutation_type]]

            while len(mutations) < PERMUT_OFF:
                temp = np.random.normal(scale=0.1, loc=-0.1)
                #     print(temp)

                temp = np.log10(mutations[0]) + temp

                if -5.0 <= temp:
                    if temp >= 1.0:
                        temp = 1.0
                else:
                    temp = -5.0

                lr = 10**temp

                mutations.append(lr)

        elif mutation_type == "DROPOUT":
            mutation_type = np.random.choice(["DO_A", "DO_B", "DO_C", "DO_D", "DO_E", "DO_F"])

            mutations = [params[mutation_type]]

            while len(mutations) < PERMUT_OFF:
                temp = np.random.normal(scale=0.2, loc=0.0)

                temp += mutations[0]

                if 0.0 <= temp <= 0.9999:
                    mutations.append(temp)

        # keep track of fitness scores so we can see which is best!
        fitness_tracker = []

        time_tracker = []
        best_acc_tracker = []

        for mutation in mutations:
            print("Mutation Type: {:}  Mutation Value: {:}".format(mutation_type, str(mutation))
            params[mutation_type] = mutation

            fitness, best_acc, best_epoch, total_time = run_model_x(params)

            fitness_tracker.append(fitness)
            time_tracker.append(total_time)
            best_acc_tracker.append(best_acc)

        best_generation = np.argmax(fitness_tracker)
        print("\nThe best mutations was", best_generation,
              "with a score of", np.max(fitness_tracker),
              "which corresponds to a value of ", mutations[best_generation], mutation_type)
        print("\n")

        # actually integrate the improvement
        params[mutation_type] = mutations[best_generation]

        log.append([np.max(fitness_tracker),
                    mutation_type,
                    best_acc_tracker[best_generation],
                    time_tracker[best_generation]])

        burn_log(log, log_file)

        print(params, "\n\n")


def burn_log(log, log_name):

    boink = pd.DataFrame(log)

    boink.columns = ["Fitness", "Mutated_Feature", "Peak_Val_Acc", "Train_Time"]

    boink.head()

    boink.to_hdf(log_name, key="log", comp_level=9)



def load_MNIST():
    '''
    Load and wrangle MNIST dataset and return X_train, X_test, y_train, y_test
    '''
    num_classes = 10

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
    input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, x_test, y_train, y_test


def run_model_x(params, x=3):
    '''
    There is variance between runs when running a model.  Therefore run a model
    x times and pick the median run, return that as the 'model results'.  Thus,
    when the genetic alg is attempting to select survivors it has a better chance
    of picking a true improvement rather than a random fluctuation.
    '''
    
    f_list = []
    ba_list = []
    be_list = []
    t_list = []
    
    for idx in range(x):
        fitness, best_acc, best_epoch, total_time = run_model(params)
        
        f_list.append(fitness)
        ba_list.append(best_acc)
        be_list.append(best_epoch)
        t_list.append(total_time)
        
    median_fitness = np.median(f_list)
    
    _index = f_list.index(median_fitness)
    
    print("\n")  # make it clear when one mutation ends and another starts
    return f_list[_index], ba_list[_index], be_list[_index], t_list[_index]


def run_model(params):
    BATCH_SIZE = params["BATCH_SIZE"]
    EPOCHS = params["EPOCHS"]
    LEARNING_RATE = params["LEARNING_RATE"]
    LR_DECAY = params["LR_DECAY"]
    MAX_GOAL_TIME = params["MAX_GOAL_TIME"]
    BEST_TIME_BONUS = params["BEST_TIME_BONUS"]
    
    start_time = time.time()

    input_img = Input(shape=(img_cols * img_rows,))
    x = Dropout(params["DO_A"])(input_img)
    x = Dense(params["NODES_FC_A"], activation=params["ACT_FC_A"])(x)
    
    x = Dropout(params["DO_B"])(x)
    x = Dense(params["NODES_FC_B"], activation=params["ACT_FC_B"])(x)
    
    x = Dropout(params["DO_C"])(x)
    x = Dense(params["NODES_FC_C"], activation=params["ACT_FC_C"])(x)
    
    x = Dropout(params["DO_D"])(x)
    x = Dense(params["NODES_FC_D"], activation=params["ACT_FC_D"])(x)
    
    x = Dropout(params["DO_E"])(x)
    x = Dense(params["NODES_FC_E"], activation=params["ACT_FC_E"])(x)
    
    x = Dropout(params["DO_F"])(x)
    output = Dense(10, activation='softmax', name='final_softmax')(x)

    # actually define the model:
    model = Model(inputs=input_img, outputs=output)

    optimizer = params['OPTIMIZER'](lr=LEARNING_RATE, decay=LR_DECAY)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer,
                  metrics=['accuracy'])

    history = model.fit(params["x_train"], params["y_train"],
                        shuffle=True,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        verbose=0,
                        validation_data=(params["x_test"], params["y_test"]))

#     print(history.history['val_acc'], "\n\n")
    
    # try to free model to prevent "time elapsed" increasing problem. (the more models run the longer the next one takes)
    del model
    
    best_acc = np.max(history.history['val_acc']) * 100
    best_epoch = np.argmax(history.history['val_acc']) + 1
    total_time = time.time() - start_time
    score_time = best_epoch / len(history.history['val_acc'])
#     print("Adjusted Score time {:3.2f}".format(total_time * score_time))
    adjustment_multiplier = 1.0 + ((MAX_GOAL_TIME - (total_time * score_time)) / MAX_GOAL_TIME)
    
    # prevent it from only prioritizing speed
    if adjustment_multiplier > BEST_TIME_BONUS:
        adjustment_multiplier = BEST_TIME_BONUS
        
    fitness = best_acc * adjustment_multiplier

    print("{:03.3f} <--Overall Fitness Score Best Accuracy: {:03.2f} in Epoch: {:} Time Elapsed: {:03.2f}s".format(fitness,
                                                                                                               best_acc,
                                                                                                               best_epoch,
                                                                                                               total_time))
    
    return fitness, best_acc, best_epoch, total_time


if __name__ == "__main__":
    # load up data
    x_train, x_test, y_train, y_test = load_MNIST()
    
    m_params["x_train"] = x_train
    m_params["x_test"] = x_test
    m_params["y_train"] = y_train
    m_params["y_test"] = y_test
    
    darwin(m_params, 50)