import numpy as np
import tensorflow as tf
import cv2

def discount_rewards(rewards, gamma=0.99):
    """Discounts an array of rewarwds, generally used after
    an episode ends and before training.

    Args:
        rewards (:obj:`list` of float): The rewards to be discounted.
        gamma (float, optional): Gamma in the reward discount function.
            Higher gamma = higher importance on later rewards.

    Returns:
        (:obj:`list` of float): List of the disounted rewards.

    Examples:
        >>> print([round(x) for x in discount_rewards([1, 2, 3], gamma=0.99)])
            [5.92, 4.97, 3.0]
    """
    new_rewards = [float(rewards[-1])]
    for i in reversed(range(len(rewards)-1)):
        new_rewards.append(float(rewards[i]) + gamma * new_rewards[-1])
    return new_rewards[::-1]

def reshape_train_var(var):
    """Reshape a list in the form [num_samples, single_var_shape],
    so that it is ready for training.

    Args:
        var (:obj:`list`): A list of training data that has a length of the number of samples.

    Returns:
        (:obj:`np.array`): Same input data with shape, [num_samples, single_var_shape].

    Examples:
        >>> print(reshape_train_var([np.array([1, 2, 3]), np.array([4, 5, 6])]))
            [[1, 2, 3]
             [4, 5, 6]]
             
        >>> print(reshape_train_var([np.array([1, 2, 3]), np.array([4, 5, 6])]).shape)
            (2, 3)
    """
    var_shape = np.array(var[0]).shape
    if var_shape == ():
        return var
    n_samples = len(var)
    concated = np.concatenate(var)
    reshaped = np.array(concated).reshape([n_samples] + list(var_shape))
    return reshaped

def gaussian_likelihood(x, mu, std):
    """Calculates the log probability of a gaussian given some input.

    Args:
        x: 2D tensor for observations drawn from the gaussian distributions.
        mu: 1D tensor for mean of the gaussian distributions.
        std: Tensor scalar for standard deviation of the gaussian distributions.

    Returns:
        1D tensor with a length equal to the number of rows in x.
        Gives the gaussian likelihood for each x for the respective gaussian distribution.

    Examples:
        >>> print([round(x) for x in discount_rewards([1, 2, 3], gamma=0.99)])
            [5.92, 4.97, 3.0]
    """
    pre_sum = -(0.5*tf.log(2.*np.pi)) - (0.5*tf.log(std)) - (tf.square(x - mu))/(2.*std+1e-8)
    
    return tf.reduce_sum(pre_sum, axis=1)

def to_grayscale(img):
    """Returns the grayscale of a numpy array with rgb channels"""
    return np.dot(img[:,:,:3], [0.299, 0.587, 0.114])

def resize_img(img, size=(84, 84)):
    """Resize the img to the shape (size_1, size_2, 1)"""
    return cv2.resize(img, dsize=size).reshape(list(size) + [1])

def scale_color(img, max_val=255.):
    """Scales an image of color values (int) [0,255] to (float) [0,1]"""
    return img / max_val

def preprocess_atari(obs, size=(84, 84)):
    """Applies grayscale, resize, and scale_color to one observation, returning the new img"""
    gray = to_grayscale(obs)
    resized = resize_img(gray, size)
    return scale_color(resized)
