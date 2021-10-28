
import unittest
from genetic import find_best, cross_points, cross

class TestGenetic(unittest.TestCase):
    
    def test_find_best(self):
        population = [(2, 6), (-2, 5), (10, 0)]
        ratings = [10, 4, -2]
        self.assertEqual(find_best(population, ratings), ((10, 0), -2))
        population = [(2, 6), (-2, 5), (10, 0)]
        ratings = [-2, 4, -2]
        self.assertEqual(find_best(population, ratings), ((2, 6), -2))        
        
        population = [(2, 6)]
        ratings = [10, 4]
        self.assertRaises(ValueError, find_best, population, ratings)
        population = []
        ratings = []
        self.assertRaises(ValueError, find_best, population, ratings)

    def test_cross(self):
        a = -1 # 11111
        b = -12 # 10100
        self.assertTrue(cross(a, b) in [[-4, -9], [-12, -1], [-4, -9], [-2, -11]])


if __name__ == '__main__':
    unittest.main()