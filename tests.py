import os
import unittest

from cobs.run_model import fingerprint
from src.models.keras_models import Residual, Perceptron, RNN, GRU, MultilayerPerceptron
from keras.models import Sequential

default_seq = "tactagcaatacgcttgcgttcggtggttaagtatgtataatgcgcgggcttgtcgt"
default_len = 10

input_shape = 256
output_shape = 2
input_length = 128


class TestModels(unittest.TestCase):
    def test_build_residual(self):
        model = Residual(input_shape, output_shape)
        self.assertIsInstance(model, Sequential)

    def test_build_perceptron(self):
        model = Perceptron(input_shape, output_shape)
        self.assertIsInstance(model, Sequential)

    def test_build_rnn(self):
        model = RNN(input_shape, output_shape, input_length)
        self.assertIsInstance(model, Sequential)

    def test_build_gru(self):
        model = GRU(input_shape, output_shape, input_length)
        self.assertIsInstance(model, Sequential)

    def test_build_mperceptron(self):
        model = MultilayerPerceptron(input_shape, output_shape)
        self.assertIsInstance(model, Sequential)


class TestFingerprint(unittest.TestCase):
    def output_fingerprint(self):
        output = fingerprint(default_seq, default_len)
        self.assertIsInstance(output, list)

    def output_len(self):
        output = fingerprint(default_seq, default_len)
        self.assertEqual(default_len, len(output))


if __name__ == '__main__':
    unittest.main()
