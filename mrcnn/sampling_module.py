import numpy as np
import tensorflow as tf
import keras.layers as KL
import keras.models as KM
import keras.engine as KE


class SelectLayer(KE.Layer):
    """
    select a subset of the sample points with the sample stride
    """

    def __init__(self, image_shape, stride, **kwargs):
        """
        :param image_shape: the shape of the original image
        :param stride: the sample stride
        """
        super(SelectLayer, self).__init__(**kwargs)
        self.image_shape = image_shape
        self.stride = stride
        self.trainable = False

    def call(self, inputs):
        index_tensor = inputs
        # get the index of sample points that can be divided by the stride
        mask = tf.logical_not(tf.cast(tf.reduce_sum(tf.mod(index_tensor, self.stride), axis=-1), dtype=tf.bool))
        # get the points that the corresponding mask is true
        result = tf.cast(tf.boolean_mask(index_tensor, mask), tf.int32)
        # scale the points from the original image to the feature map
        half_w = self.image_shape[0] // 2
        result = (result - half_w) // self.stride + (self.image_shape[0] // self.stride) // 2
        result = tf.cast(result, tf.int32)
        return result


class ExpandLayer(KE.Layer):
    """
    expand the points to the batch size to match every feature map in a batch
    """

    def __init__(self, batch_size, **kwargs):
        """
        :param batch_size: the batch size
        """
        super(ExpandLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.trainable = False

    def call(self, inputs):
        """
        :param inputs: the selected index for the feature map
        expand the points to the batch size to match every feature map in a batch.
        For example: the select_index=[[2,2],[3, 4]], batch_size=4,
        the output is: [[[0, 2, 2],[0, 3, 4]],[[1, 2, 2],[1, 3, 4]],[[2, 2, 2],[2, 3, 4]],[[3, 2, 2],[3, 3, 4]]]
        """
        select_index = inputs
        shape = tf.shape(select_index)
        batch_index = tf.range(0, self.batch_size, dtype=tf.int32)
        batch_index = tf.expand_dims(batch_index, axis=1)
        batch_index = tf.tile(batch_index, (1, shape[0]))
        batch_index = tf.expand_dims(batch_index, axis=-1)
        select_index = tf.broadcast_to(select_index, shape=tf.concat([tf.constant([self.batch_size]), shape], axis=0))
        result = tf.concat([batch_index, select_index], axis=-1)
        return result


def sampling_graph(sample_points, stride, image_shape, batch_size, name):
    """
    combine the select layer and expand layer to build a graph
    :param sample_points: the index that will be used to generate anchor on the original image
    :param stride: the sampling stride
    :param image_shape: the image shape
    :param batch_size: the batch size
    :param name: the name of the output layer
    """
    select = SelectLayer(image_shape=image_shape, stride=stride)(sample_points)
    expand = ExpandLayer(batch_size=batch_size, name=name)(select)
    return expand


class GatherLayer(KE.Layer):
    """
    select the pixel that will be used to generate anchor on the feature map
    """
    def __init__(self, batch_size, depth, **kwargs):
        """
        :param batch_size: batch size
        :param depth:  the channel of the feature map
        """
        super(GatherLayer, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.depth = depth

    def call(self, inputs, **kwargs):
        feature_map = inputs[0]
        ids = inputs[1]
        # select the pixel that will be used to generate anchor on the feature map
        gauss_feature = tf.gather_nd(feature_map, ids, name="feature_with_gauss_ids")
        # flatten selected pixels to facilitate subsequent calculations
        gauss_feature = tf.reshape(gauss_feature, shape=(self.batch_size, 1, -1, self.depth),
                                   name="feature_with_gauss_ids_reshape")
        return gauss_feature


class SelfLayer(KE.Layer):
    def __init__(self, **kwargs):
        super(SelfLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs[0]
        y = inputs[1]
        z = tf.boolean_mask(x, y)
        return z
