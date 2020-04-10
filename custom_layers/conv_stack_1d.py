
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import MaxPool1D, BatchNormalization, Conv1D
from tensorflow.keras import regularizers

from .util import pipe_model

def conv_stack_1d(filters, kernels, strides, max_pool_sizes, batch_norms=0, padding="same", activation="elu", l2=0,
    cross_activation_lambda=0, activation_lambda=0, names=None):
    """Represents a stack of convolutional layers"""

    # If padding is one word or default, extend into a uniform list
    if type(batch_norms) != list:
        batch_norms = [batch_norms]*len(filters)

    # If padding is one word or default, extend into a uniform list
    if type(padding) != list:
        padding = [padding]*len(filters)

    # If activation is one word or default, extend into a uniform list
    if type(activation) != list:
        activation = [activation]*len(filters)

    # If names is one word or default, extend into a uniform list
    if type(names) != list:
        names = [names]*len(filters)

    layers = []
    for i in range(len(filters)):

        layers.append(Conv1D(
                        filters=filters[i],
                        kernel_size=kernels[i],
                        strides=strides[i],
                        padding=padding[i],
                        activation=activation[i],
                        kernel_regularizer=regularizers.l2(l2/filters[i]),
                        activity_regularizer=regularizers.l1(activation_lambda/filters[i]),
                        name=names[i]
            ))

        if cross_activation_lambda > 0:
            layers.append(CrossActivationRegularization(n_filters=filters[i], cross_activation_lambda=cross_activation_lambda))

        if batch_norms[i] == 1:
            layers.append(BatchNormalization(axis=2))

        if len(max_pool_sizes) != 0:
            layers.append(MaxPool1D(pool_size=max_pool_sizes[i]))

    def conv_stack_1d_layer(inputs):
        """Layer hook for stack"""

        return pipe_model(inputs, layers)

    return conv_stack_1d_layer

def get_raw_kernel_l2(weight_matrix):
    """Simple l2 computation on weight matrix"""

    kernel_l2 = tf.math.square(weight_matrix)
    return tf.reduce_sum(kernel_l2)

def get_raw_activity_l1(activation_matrix):
    """Simple l1 computation on activation matrix"""

    activity_l1 = tf.math.abs(activation_matrix)
    activity_l1 = tf.reduce_sum(activity_l1) / activation_matrix.shape[2]
    return activity_l1


class CrossActivationRegularization(tf.keras.layers.Layer):
    """
    Encourages model to find different patterns by penalizing correlation of
    any sort (positive or negative) between filter activations.

    Creates metrics out of cross activation loss and activity l1 loss
    """

    def __init__(self, n_filters, cross_activation_lambda=0, name=""):
        if name != "":
            super().__init__(name=name)
        else:
            super().__init__()

        self.n_filters = n_filters
        self.cross_activation_lambda = cross_activation_lambda

    def call(self, inputs):

        # Compute correlation for each batch sample, and average across batch
        corr_avg_abs = self.get_avg_abs_activation_correlation(inputs)

        # Sum the upper triangular region
        tri_sum = self.get_upper_triangular_sum(corr_avg_abs)

        # Add loss to layer
        self.add_loss(tri_sum * self.cross_activation_lambda / self.get_unique_filter_pairs())

        # Monitor raw cross activation loss during training
        self.add_metric(tri_sum, name="cross_activation_loss",
            aggregation="mean")

        # Monitor raw activation l1 during training
        self.add_metric(get_raw_activity_l1(inputs), name="activity_l1_loss",
            aggregation="mean")

        return inputs

    def get_unique_filter_pairs(self):

        n_unique_pairs = 0
        for i in range(self.n_filters-1):
            n_unique_pairs += i + 1

        return n_unique_pairs

    def get_upper_triangular_sum(self, mat):
        
        # Create mask to cancel out all values except for the upper right matrix triangle (excluding diagonal)
        n_masks = 0
        for i in range(mat.shape[0]):
            n_masks += i+1
        mask = tfp.math.fill_triangular([-1]*n_masks)
        mask = tf.cast(mask, dtype=tf.float32)
        mask = mask + 1
        
        # Apply mask to matrix
        triangular = mat * mask

        # Sum the triangular
        tr_sum = tf.reduce_sum(triangular)

        return tr_sum

    def get_avg_abs_activation_correlation(self, inputs):

        # Create loop to find the correlation of every filter vector
        i = tf.constant(0)
        corr_sum = tf.constant(0, shape=(self.n_filters, self.n_filters), dtype=tf.float32)

        def cond(i, corr_sum):
            return tf.less(i, tf.shape(inputs)[0])

        def body(i, corr_sum):
            corr_sum = corr_sum + tfp.stats.correlation(inputs[i], inputs[i], sample_axis=0, event_axis=1)
            return tf.add(i, 1), corr_sum

        # Run loop
        i, corr_sum = tf.while_loop(cond, body, [i, corr_sum])

        # Average across batch size and take absolute value
        corr_avg = corr_sum / tf.cast(tf.shape(inputs)[0], dtype=tf.float32)
        corr_avg_abs = tf.math.abs(corr_avg)

        return corr_avg_abs







