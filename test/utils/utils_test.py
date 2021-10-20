import unittest
import numpy as np
import torch

from src.utils.utils import MSELoss, RouletteWheelSelector

class TestRouletteWheel(unittest.TestCase):
    def setUp(self) -> None:
        self.wheel = RouletteWheelSelector()
        np.random.seed(42)
        return super().setUp()

    def test_one_probabilities(self):
        selected = self.wheel.select(np.array([0, 0, 0, 1]))
        np.testing.assert_array_equal(np.array([3, 3, 3, 3]), selected)
    
    def test_zero_probs_are_not_contained(self):
        selected = self.wheel.select(np.array([0, 1, 1, 0]))
        np.testing.assert_array_equal(np.array([1, 2, 2, 2]), selected)

    def test_correct_shape(self):
        given = np.array([1.0, 0.5, 0.5, 3.0, 5.0])
        selection = self.wheel.select(given)
        self.assertEquals(given.shape, selection.shape)

class TestMSE(unittest.TestCase):
    def test_mse_iszero(self):
        y_pred, y_true = torch.tensor(1.0), torch.tensor(1.0)
        loss = MSELoss(y_true, y_pred)
        self.assertEquals(torch.tensor(0), loss)

    def test_large_sided_triangle(self):
        y_pred, y_true = torch.tensor(10.0), torch.tensor(1.0)
        loss = MSELoss(y_pred, y_true)
        self.assertEquals(torch.tensor(81.0), loss)


if __name__ == '__main__':
    unittest.main()