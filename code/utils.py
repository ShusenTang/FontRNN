# -*- coding: utf-8 -*-
import random
import numpy as np

# A note on formats:
# Character is encoded as a sequence of strokes. stroke-3 and stroke-5 are
# different stroke encodings.
#   stroke-3 uses 3-tuples, consisting of x-offset, y-offset, and a binary
#       variable which is 1 if the pen is lifted between this position and
#       the next, and 0 otherwise.
#   stroke-5 consists of x-offset, y-offset, and p_1, p_2, p_3, a binary
#   one-hot vector of 3 possible pen states: pen down, pen up(end of stroke), 
#   end of character. See section 3.1 of paper for more detail.
# FontRNN takes input in stroke-5 format, with strokes padded to a common
# maximum length and prefixed by the special start token [0, 0, 1, 0, 0]
# The raw dataset is stored using stroke-3.


def augment_strokes(strokes, prob=0.0):
    """Perform data augmentation by randomly dropping out strokes."""
    # drop each point within a line segments with a probability of prob
    # note that the logic in the loop prevents points at the ends to be dropped.
    result = []
    prev_stroke = [0, 0, 1]
    count = 0
    stroke = [0, 0, 1]  # Added to be safe.
    for i in range(len(strokes)):
        candidate = [strokes[i][0], strokes[i][1], strokes[i][2]]
        if candidate[2] == 1 or prev_stroke[2] == 1:
            count = 0
        else:
            count += 1
        urnd = np.random.rand()  # uniform random variable
        if candidate[2] == 0 and prev_stroke[2] == 0 and count > 2 and urnd < prob:
            stroke[0] += candidate[0]
            stroke[1] += candidate[1]
        else:
            stroke = candidate
            prev_stroke = stroke
            result.append(stroke)
    return np.array(result)


def to_normal_strokes(big_stroke):
    """Convert from stroke-5 format back to stroke-3."""
    l = 0
    for i in range(len(big_stroke)):
        if big_stroke[i, 4] > 0:
            l = i
            break
    if l == 0:
        l = len(big_stroke)
    result = np.zeros((l, 3))
    result[:, 0:2] = big_stroke[0:l, 0:2]
    result[:, 2] = big_stroke[0:l, 3]
    return result


def to_big_strokes(stroke, max_len=250):
    """Converts from stroke-3 to stroke-5 format and pads to given length."""
    # (But does not insert special start token).
    result = np.zeros((max_len, 5), dtype=float)
    l = len(stroke)
    assert l <= max_len
    result[0:l, 0:2] = stroke[:, 0:2]
    result[0:l, 3] = stroke[:, 2]
    result[0:l, 2] = 1 - result[0:l, 3]
    result[l:, 4] = 1 # p3 = 1 if and only if i > l
    return result


def get_max_len(strokes):
    """Return the maximum length of an array of strokes."""
    max_len = 0
    for stroke in strokes:
        ml = len(stroke)
        if ml > max_len:
            max_len = ml
    return max_len


class DataLoader(object):
    """Class for loading data."""
    def __init__(self,
                 strokes,
                 batch_size=100,
                 max_seq_length=200,
                 random_scale_factor=0.0,
                 augment_stroke_prob=0.0):
        self.strokes = strokes
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length 
        self.random_scale_factor = random_scale_factor  # data augmentation method
        self.augment_stroke_prob = augment_stroke_prob  # data augmentation method
        self.start_stroke_token = [0, 0, 1, 0, 0]  # T_0 in paper
        self.num_batches = int(len(strokes) / self.batch_size)
        self.scale_factor = None


    def random_scale(self, data):
        """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
        x_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        y_scale_factor = (np.random.random() - 0.5) * 2 * self.random_scale_factor + 1.0
        result = np.copy(data)
        result[:, 0] *= x_scale_factor
        result[:, 1] *= y_scale_factor
        return result


    def normalize(self, scale_factor):
        """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
        if self.scale_factor is None:
            self.scale_factor = scale_factor
        assert self.scale_factor == scale_factor
        for i in range(len(self.strokes)):
            self.strokes[i][:, 0:2] /= self.scale_factor

    def _get_batch_from_indices(self, indices):
        """Given a list of indices, return the potentially augmented batch."""
        x_batch = []
        seq_len = []
        for idx in range(len(indices)):
            i = indices[idx]
            data = self.random_scale(self.strokes[i])
            data_copy = np.copy(data)
            if self.augment_stroke_prob > 0:
                data_copy = augment_strokes(data_copy, self.augment_stroke_prob)
            x_batch.append(data_copy)
            length = len(data_copy)
            seq_len.append(length)
        seq_len = np.array(seq_len, dtype=int)
        # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
        return x_batch, self.pad_batch(x_batch, self.max_seq_length), seq_len

    def get_batch(self, idx):
        """Get the idx'th batch from the dataset."""
        assert idx >= 0 # idx must be non negative
        assert idx < self.num_batches # idx must be less than the number of batches
        start_idx = idx * self.batch_size
        indices = range(start_idx, start_idx + self.batch_size)
        return self._get_batch_from_indices(indices)

    def pad_batch(self, batch, max_len):
        """Pad the batch to be stroke-5 bigger format as described in paper."""
        result = np.zeros((self.batch_size, max_len + 1, 5), dtype=float)
        assert len(batch) == self.batch_size
        for i in range(self.batch_size):
            l = len(batch[i])
            assert l <= max_len
            result[i, 0:l, 0:2] = batch[i][:, 0:2]
            result[i, 0:l, 3] = batch[i][:, 2]
            result[i, 0:l, 2] = 1 - result[i, 0:l, 3]
            result[i, l:, 4] = 1

            result[i, 1:, :] = result[i, :-1, :]
            result[i, 0, :] = 0
            result[i, 0, 2] = self.start_stroke_token[2]  # setting T_0
            result[i, 0, 3] = self.start_stroke_token[3]
            result[i, 0, 4] = self.start_stroke_token[4]
        return result
