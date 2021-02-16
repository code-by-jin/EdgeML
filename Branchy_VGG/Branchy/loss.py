import tensorflow as tf

def cross_entropy(logits, labels):
    '''
    This cross entropy is used for logits, NOT output from softmax layer
    And the groug truth labels are single numbers but NOT one-hot coded format
    '''
    labels = tf.cast(labels, tf.int32)
    cross_entropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,                                                               name='cross_entropy_per_example')
    #cross_entropy_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name=labels)
    cross_entropy_mean = tf.reduce_mean(cross_entropy_, name='cross_entropy')
    return cross_entropy_mean

def cross_entropy_softmax(softmax, labels, num_class):
    '''
    This loss cross entropy loss is used for softmax output
    The input lables is single number, it should be converted into one-hot code
    '''
    #labels = tf.cast(labels, tf.int32)
    labels = tf.one_hot(labels, num_class)
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(softmax), reduction_indices=[1]))
    return cross_entropy


def top_k_error(predictions, labels, k=1, test=False):
    if test:
        batch_size = 1
    else:
        batch_size = predictions.get_shape().as_list()[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k))
    num_correct = tf.reduce_sum(in_top1)
    return num_correct / float(batch_size)
