import unittest

# Import from sibling package; assumes tests run from repo root
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import compute_relevance_score


class TestComputeRelevanceScore(unittest.TestCase):
    def test_parallel_vectors(self):
        u = [1.0, 0.0, 0.0]
        v = [2.0, 0.0, 0.0]
        score = compute_relevance_score(u, v, normalize=True)
        self.assertAlmostEqual(score, 1.0, places=7)

    def test_orthogonal_vectors(self):
        u = [1.0, 0.0]
        v = [0.0, 1.0]
        score = compute_relevance_score(u, v, normalize=True)
        self.assertAlmostEqual(score, 0.5, places=7)

    def test_opposite_vectors(self):
        u = [1.0, 0.0]
        v = [-1.0, 0.0]
        score = compute_relevance_score(u, v, normalize=True)
        self.assertAlmostEqual(score, 0.0, places=7)

    def test_zero_norm_fallback(self):
        u = [0.0, 0.0]
        v = [1.0, 0.0]
        score = compute_relevance_score(u, v, normalize=True)
        self.assertTrue(0.0 <= score <= 1.0)


if __name__ == "__main__":
    unittest.main()


