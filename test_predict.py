import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.predict_pipeline import PredictPipeline

def test_inference():
    pipeline = PredictPipeline()
    res = pipeline.predict(recency=10, frequency=12, monetary=2500.0)
    print("Inference Result:", res)
    assert res["cluster_id"] in [0, 1, 2, 3, 4, 5, 6]
    assert res["persona"] != ""
    print("Inference test PASSED!")

if __name__ == "__main__":
    test_inference()
