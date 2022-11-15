import tensorflow as tf
import numpy as np
from warprnnt_tensorflow import rnnt_loss

class WarpRNNTTest(tf.test.TestCase):

    def _run_rnnt(self, acts, labels, input_lengths, label_lengths,
                  expected_costs, expected_grads, blank, fastemit_lambda, monotonic, use_gpu=False):
        self.assertEqual(acts.shape, expected_grads.shape)

        acts_t = tf.constant(acts)
        labels_t = tf.constant(labels)
        input_lengths_t = tf.constant(input_lengths)
        label_lengths_t = tf.constant(label_lengths)

        with tf.GradientTape() as tape:
            tape.watch(acts_t)
            logits = acts_t if use_gpu else tf.nn.log_softmax(acts_t)
            costs = rnnt_loss(logits, labels_t, input_lengths_t, label_lengths_t, blank, fastemit_lambda, monotonic)
        grads = tape.gradient(costs, [acts_t])[0]

        #self.assertAllClose(costs, expected_costs, atol=1e-6)
        self.assertAllClose(grads, expected_grads, atol=1e-6)

    def test_forward(self):
        # Softmax activations for the following inputs:
        acts = np.array([0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                        0.1, 0.6, 0.1, 0.1, 0.1, 0.1,
                        0.2, 0.8, 0.1, 0.1, 0.6, 0.1,
                        0.1, 0.1, 0.1, 0.1, 0.2, 0.1,
                        0.1, 0.7, 0.1, 0.2, 0.1, 0.1], dtype=np.float32).reshape(1, 2, 3, 5)

        labels = np.array([[1, 2]], dtype=np.int32)
        input_lengths = np.array([2], dtype=np.int32)
        label_lengths = np.array([2], dtype=np.int32)

        acts_t = tf.constant(acts)
        labels_t = tf.constant(labels)
        input_lengths_t = tf.constant(input_lengths)
        label_lengths_t = tf.constant(label_lengths)

        acts_t = tf.nn.log_softmax(acts_t) # NOTE cpu

        costs = rnnt_loss(acts_t, labels_t, input_lengths_t, label_lengths_t)
        print(costs)

    def _test_multiple_batches(self, use_gpu):
        B = 2; T = 4; U = 3; V = 3

        acts = np.array([0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135,
                        0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539,
                        0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882,
                        0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400,
                        0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106,
                        0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160,
                        0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467,
                        0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474,
                        0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356,
                        0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830,
                        0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289,
                        0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021], dtype=np.float32).reshape(B, T, U, V);

        expected_costs = np.array([4.28065, 3.93844], dtype=np.float32)
        expected_grads = np.array([-0.186844, -0.062555, 0.249399, -0.203377, 0.202399, 0.000977,
                                    -0.141016, 0.079123, 0.061893, -0.011552, -0.081280, 0.092832,
                                    -0.154257, 0.229433, -0.075176, -0.246593, 0.146405, 0.100188,
                                    -0.012918, -0.061593, 0.074512, -0.055986, 0.219831, -0.163845,
                                    -0.497627, 0.209240, 0.288387, 0.013605, -0.030220, 0.016615,
                                    0.113925, 0.062781, -0.176706, -0.667078, 0.367659, 0.299419,
                                    -0.356344, -0.055347, 0.411691, -0.096922, 0.029459, 0.067463,
                                    -0.063518, 0.027654, 0.035863, -0.154499, -0.073942, 0.228441,
                                    -0.166790, -0.000088, 0.166878, -0.172370, 0.105565, 0.066804,
                                    0.023875, -0.118256, 0.094381, -0.104707, -0.108934, 0.213642,
                                    -0.369844, 0.180118, 0.189726, 0.025714, -0.079462, 0.053748,
                                    0.122328, -0.238789, 0.116460, -0.598687, 0.302203, 0.296484], dtype=np.float32).reshape(B, T, U, V);

        labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
        input_lengths = np.array([4, 4], dtype=np.int32)
        label_lengths = np.array([2, 2], dtype=np.int32)

        self._run_rnnt(acts, labels, input_lengths, label_lengths, expected_costs, expected_grads, 0, 0.0, monotonic=False, use_gpu=use_gpu)

    def _test_multiple_batches_fastemit(self, use_gpu):
        B = 2; T = 4; U = 3; V = 3

        acts = np.array([0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135,
                        0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539,
                        0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882,
                        0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400,
                        0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106,
                        0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160,
                        0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467,
                        0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474,
                        0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356,
                        0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830,
                        0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289,
                        0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021], dtype=np.float32).reshape(B, T, U, V);

        expected_costs = np.array([4.28065, 3.93844], dtype=np.float32)
        expected_grads = np.array(
            [
                 -0.04752224, -0.3434786 ,  0.3910008 , -0.14560838,  0.27445215, -0.12884367,
                 -0.14101607,  0.07912346,  0.06189261,  0.08337539, -0.23355475,  0.15017931,
                 -0.06381905,  0.31584963, -0.25203055, -0.2465931 ,  0.14640461,  0.10018849,
                  0.01392107, -0.13943397,  0.12551293,  0.02527768,  0.36032543, -0.38560316,
                 -0.497627  ,  0.20923999,  0.28838703,  0.02720971, -0.06043926,  0.03322954,
                  0.22784898,  0.12556238, -0.35341135, -0.66707844,  0.36765894,  0.29941946,
                 -0.25421223, -0.27434072,  0.52855283, -0.06629672, -0.02493926,  0.091236,
                 -0.06351757,  0.02765449,  0.03586307, -0.04519127, -0.28051704,  0.3257084,
                 -0.11377359, -0.10745317,  0.22122678, -0.17236964,  0.10556533,  0.06680433,
                  0.12432144, -0.28644192,  0.1621205 , -0.00627744, -0.3230167 ,  0.32929417,
                 -0.36984423,  0.1801181 ,  0.18972616,  0.05142741, -0.15892352,  0.10749611,
                  0.24465652, -0.47757745,  0.23292094, -0.59868705,  0.3022032 ,  0.29648384,
            ], dtype=np.float32).reshape(B, T, U, V);

        labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
        input_lengths = np.array([4, 4], dtype=np.int32)
        label_lengths = np.array([2, 2], dtype=np.int32)

        self._run_rnnt(acts, labels, input_lengths, label_lengths, expected_costs, expected_grads, 0, 1.0, monotonic=False, use_gpu=use_gpu)

    def _test_multiple_batches_monotonic(self, use_gpu):
        B = 2; T = 4; U = 3; V = 3

        acts = np.array([0.065357, 0.787530, 0.081592, 0.529716, 0.750675, 0.754135,
                        0.609764, 0.868140, 0.622532, 0.668522, 0.858039, 0.164539,
                        0.989780, 0.944298, 0.603168, 0.946783, 0.666203, 0.286882,
                        0.094184, 0.366674, 0.736168, 0.166680, 0.714154, 0.399400,
                        0.535982, 0.291821, 0.612642, 0.324241, 0.800764, 0.524106,
                        0.779195, 0.183314, 0.113745, 0.240222, 0.339470, 0.134160,
                        0.505562, 0.051597, 0.640290, 0.430733, 0.829473, 0.177467,
                        0.320700, 0.042883, 0.302803, 0.675178, 0.569537, 0.558474,
                        0.083132, 0.060165, 0.107958, 0.748615, 0.943918, 0.486356,
                        0.418199, 0.652408, 0.024243, 0.134582, 0.366342, 0.295830,
                        0.923670, 0.689929, 0.741898, 0.250005, 0.603430, 0.987289,
                        0.592606, 0.884672, 0.543450, 0.660770, 0.377128, 0.358021], dtype=np.float32).reshape(B, T, U, V);

        expected_costs = np.array([3.069, 3.2272], dtype=np.float32)
        expected_grads = np.array(
            [
                0.0063176905, -0.25571644, 0.24939875, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.084992185,
                -0.13633762, 0.051345434, -0.14593755, 0.2759514, -0.13001384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.16883233, 0.29189086, -0.46072322, -0.2122688, 0.08925384, 0.123014964, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, -0.66707826, 0.36765885, 0.2994194, -0.07102895, -0.3406622, 0.4116911,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15444079, -0.2918697, 0.13742894, -0.09298043,
                -0.10135508, 0.19433554, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.20763135, -0.45159328, 0.24396195,
                -0.17744718, 0.08641868, 0.09102852, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.59868693, 0.30220315, 0.29648378

            ], dtype=np.float32).reshape(B, T, U, V);

        labels = np.array([[1, 2], [1, 1]], dtype=np.int32)
        input_lengths = np.array([4, 4], dtype=np.int32)
        label_lengths = np.array([2, 2], dtype=np.int32)

        self._run_rnnt(acts, labels, input_lengths, label_lengths, expected_costs, expected_grads, 0, 0.0, monotonic=True, use_gpu=use_gpu)

    def test_multiple_batches_cpu(self):
        self._test_multiple_batches(use_gpu=False)

    def test_multiple_batches_cpu_fastemit(self):
        self._test_multiple_batches_fastemit(use_gpu=False)

    def test_multiple_batches_gpu(self):
        if tf.test.is_gpu_available():
            self._test_multiple_batches(use_gpu=True)
        else:
            print('Skipping GPU test, no gpus available')

    def test_multiple_batches_gpu_fastemit(self):
        if tf.test.is_gpu_available():
            self._test_multiple_batches_fastemit(use_gpu=True)
        else:
            print('Skipping GPU test, no gpus available')

    def test_multiple_batches_gpu_monotonic(self):
        if tf.test.is_gpu_available():
            self._test_multiple_batches_monotonic(use_gpu=True)
        else:
            print('Skipping GPU test, no gpus available')


if __name__ == '__main__':
    tf.test.main()
