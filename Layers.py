import tensorflow as tf

class GetPatches( tf.keras.layers.Layer ):
    def __init__(self, num_patches):
        super(GetPatches, self).__init__()
        self.num_patches = tf.constant(num_patches, dtype=tf.float32)

    def build(self, input_shape):
        self.batch_size = input_shape[0]
        self.patch_size = tf.cast(tf.math.divide(input_shape[1], tf.math.sqrt(self.num_patches)), dtype=tf.int32)
        #self.total_num_patches = tf.cast(tf.multiply(input_shape[0], self.num_patches), dtype=tf.int32)

    def call(self, inputs):
        sizes = [1, self.patch_size, self.patch_size, 1]
        patches = tf.image.extract_patches(images=inputs,
                                     sizes=sizes,
                                     strides=sizes,
                                     rates=[1, 1, 1, 1],
                                     padding='VALID')
        return tf.reshape(patches, [self.batch_size, tf.cast(self.num_patches, dtype=tf.int32), self.patch_size, self.patch_size, 1])

class GetBatchCosts(tf.keras.layers.Layer):
    def __init__(self, ring_size=1, patches=False):
        super(GetBatchCosts, self).__init__()
        self.ring_size = tf.constant(ring_size, dtype=tf.float64)
        self.patches = patches

    def sum_rings(self, inputs):
        inputs_shape = tf.shape(inputs)
        inputs = tf.reshape(inputs, [inputs_shape[0], inputs_shape[1], inputs_shape[2]])
        inputs_shape = tf.shape(inputs)

        # input_shape.shape[1] == input_shape.shape[2]
        tf.assert_equal(inputs_shape[1], inputs_shape[2], message='Cannot create rings for non-square fourier image.')

        # math.ceil(input_shape.shape[1] / 2) % self.ring_size == 0
        tf.assert_equal(tf.math.mod(tf.math.ceil(tf.math.truediv(inputs_shape[1], 2)), self.ring_size),
                        tf.constant(0, dtype=tf.float64),
                        message='Height of fourier image is not evenly divisible into rings')
        # math.ceil(input_shape.shape[2] / 2) % self.ring_size == 0
        tf.assert_equal(tf.math.mod(tf.math.ceil(tf.math.truediv(inputs_shape[2], 2)), self.ring_size),
                        tf.constant(0, dtype=tf.float64),
                        message='Width of fourier image is not evenly divisible into rings')

        tf.assert_equal(inputs_shape[1], inputs_shape[2],
                        message='Cannot create rings for non-square fourier image.')

        # The sparse matrix will not track zero values returned by the fourier transform
        # so we add a small value to each element after the fourier transform
        tf_fft = tf.abs(tf.signal.fftshift(tf.signal.fft2d(tf.cast(inputs, dtype=tf.complex64)))) + tf.constant(0.001)
        tf_fft_shape = tf.shape(tf_fft)
        ef = tf.reshape(tf_fft, [tf_fft_shape[0], tf_fft_shape[1], tf_fft_shape[2]])
        st = tf.sparse.from_dense(ef)

        # Transform sparse matrix indices into batches format of (batch_num, image, ef_index)
        st_indices = tf.cast(tf.identity(st.indices), dtype=tf.float32)
        batch_nums, ef_idxs = tf.split(st_indices, [1, 2], axis=1)
        batched_ef_idxs = tf.reshape(ef_idxs, [inputs_shape[0], -1, 2])

        # Calculate Chebyshev distance of each pixel from the center that is (mid, mid)
        ef_shape = tf.shape(ef)
        mid_val = tf.divide(tf.subtract(ef_shape[-2], 1), 2)
        mid = tf.cast(tf.repeat(mid_val, repeats=[2]), dtype=tf.float32)
        # Only calculate distances for first image since all images are same size
        first_batched_ef_idxs = tf.expand_dims(batched_ef_idxs[0], 0)
        sub = tf.subtract(first_batched_ef_idxs, mid)
        abs_t = tf.abs(sub)
        maxxed = tf.reduce_max(abs_t, axis=2)
        floored = tf.floor(maxxed)
        floored = tf.cast(floored, dtype=tf.float64)
        div = tf.divide(floored, self.ring_size)
        dist = tf.cast(div, dtype=tf.int32)

        # Sum the pixel values of the fourier transformed image based on distance from center
        num_rings = tf.cast(tf.divide(tf.math.ceil(tf.divide(ef_shape[-2], 2)), self.ring_size), dtype=tf.int32)
        res_inputs = tf.transpose(tf.reshape(ef, [ef_shape[0], -1]))

        summed_rings = tf.transpose(tf.math.unsorted_segment_sum(res_inputs, dist[0], num_rings))
        return summed_rings

    def call(self, inputs):
        if self.patches:
            inputs_shape = tf.shape(inputs)
            return tf.reshape(tf.map_fn(self.sum_rings, inputs), [inputs_shape[0], -1])
        else:
            return self.sum_rings(inputs)

class ApplyWeights(tf.keras.layers.Layer):
    def __init__(self):
        super(ApplyWeights, self).__init__()

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[1], ),
            initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=0.3),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg(),
            name='ring_weights'
        )

    def call(self, inputs):
        return tf.multiply(inputs, self.w)

class PrintLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def call(self, inputs):
        print(tf.shape(inputs))
        return inputs
