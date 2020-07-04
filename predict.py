import sys

import pandas as pd
import numpy as np
import joblib


def predict(data) -> np.ndarray:
    '''
    This function predicts labels for data.
    data has to be np.ndarray with shape like (n_samples, n_features) 
    '''
    # load model to Python object
    return joblib.load('model.pkl').predict(data)


if __name__ == '__main__':
    # get input file's name with command line arguments 
    input_file = sys.argv[1]

    # load data to np.ndarray object
    data = pd.read_csv(input_file).values
    
    output = predict(data)

    # if output file's name exists
    if len(sys.argv) == 3:
        output_file = sys_argv[2]

        # write output to output file
        np.savetxt(output_file, output, delimiter=',')
    elif len(sys.argv) == 2:
        print(output)
    else:
        raise AttributeError('Input file\'s name was not declared.')