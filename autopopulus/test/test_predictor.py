from unittest import TestCase, main

from autopopulus.models.prediction_models import Predictor


class TestAEDitto(TestCase):
    def test_seed(self):
        predictor = Predictor(0, ["lr"], 5)


if __name__ == "__main__":
    main()
