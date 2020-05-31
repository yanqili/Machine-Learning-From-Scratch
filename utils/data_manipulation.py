import numpy as np
def divide_on_feature(X, feature_i, threshold):
    """ Divide dataset based on if sample value on feature index is larger than
        the given threshold """
    split_func = None
    #print('threshold:',threshold,' type:',type(threshold), ' isinstance(threshold, int)',isinstance(threshold, int), ' isinstance(threshold, float)',isinstance(threshold, float))
    
#    if isinstance(threshold, int) or isinstance(threshold, float) or isinstance(threshold, np.int32):
#        split_func = lambda sample: sample[feature_i] >= threshold
#    else:
#        split_func = lambda sample: sample[feature_i] == threshold
#
#    You can only make sure that when you set up a classification tree, the type passed in is a string

    if isinstance(threshold, str):
        split_func = lambda sample: sample[feature_i] == threshold
    else:
        split_func = lambda sample: sample[feature_i] >= threshold


    X_1 = np.array([sample for sample in X if split_func(sample)])
    X_2 = np.array([sample for sample in X if not split_func(sample)])

    return np.array([X_1, X_2])