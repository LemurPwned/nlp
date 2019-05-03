import tensorflow as tf


class SimpleClassifier(tf.keras.Model):
    def __init__(self, vocab_size, num_labels):
        super(SimpleClassifier, self).__init__()
        self.dense1 = tf.keras.layers.Dense(
            512, input_shape=(vocab_size), activation='relu')
        self.dense2 = tf.keras.layers.Dense((num_labels), activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)


def run_model():
    tm = SimpleClassifier(2, 10)
    tm.compile(loss='caaategorial_crossentropy',
               optimizer='adam', metrics=['accuracy'])
