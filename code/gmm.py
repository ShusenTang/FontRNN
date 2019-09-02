import tensorflow as tf
import numpy as np

INF_MIN = 1e-8


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
    """The probability distribution fo bivariate Gussian, Eq(14) in paper."""
    norm1 = tf.subtract(x1, mu1) 
    norm2 = tf.subtract(x2, mu2)
    s1s2 = tf.multiply(s1, s2)
    
    # Eq(15)
    z = (tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) -
         2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2))
    neg_rho = tf.clip_by_value(1 - tf.square(rho), INF_MIN, 1.0)
    result = tf.exp(tf.div(-z, 2 * neg_rho))
    denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(neg_rho))
    result = tf.div(result, denom)
    return result


def get_loss(pi, mu1, mu2, sigma1, sigma2, corr, pen,  pen_logits, 
                x1_data, x2_data, pen_data):
    """Loss_d(Eq16) and Loss_c(Eq17). (before dividing by L)"""

    result0 = tf_2d_normal(x1_data, x2_data, mu1, mu2, sigma1, sigma2, corr)
    # result1 is the Loss_d (without dividing by L)
    result1 = tf.multiply(result0, pi)
    result1 = tf.reduce_sum(result1, 1, keepdims=True)
    result1 = -tf.log(result1 + INF_MIN)  # avoid log(0)

    _mask = 1.0 - pen_data[:, 2]  # mask eos columns
    mask = tf.reshape(_mask, [-1, 1])
    # Zero out loss terms beyond L, the last actual stroke
    result1 = tf.multiply(result1, mask)

    result2 = tf.nn.softmax_cross_entropy_with_logits(labels=pen_data, logits=pen_logits)
    result2 = tf.reshape(result2, [-1, 1])

    return result1, result2 # Ld, Lc; before dividing by L


def get_mixture_coef(output):
    """Split the output and return the Mixture Density Network params."""
    pen_logits = output[:, 0:3]  # pen states
    pi, mu1, mu2, sigma1, sigma2, corr = tf.split(output[:, 3:], 6, 1)

    # softmax all the pi's and pen states:
    pi = tf.nn.softmax(pi) # Eq(9)
    pen = tf.nn.softmax(pen_logits) # Eq(12)

    # exponentiate the sigmas and make corr between -1 and 1.
    sigma1 = tf.exp(sigma1) # Eq(10)
    sigma2 = tf.exp(sigma2)
    corr = tf.tanh(corr) # Eq(11)

    result = [pi, mu1, mu2, sigma1, sigma2, corr, pen, pen_logits]
    return result


def sample_gaussian_2d(mu1, mu2, s1, s2, rho, sqrt_temp=1.0, greedy=False):
    if greedy:
        return mu1, mu2
    mean = [mu1, mu2]
    s1 *= sqrt_temp * sqrt_temp
    s2 *= sqrt_temp * sqrt_temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]] 
    x = np.random.multivariate_normal(mean, cov, 1)  # sample randomly
    return x[0][0], x[0][1]
