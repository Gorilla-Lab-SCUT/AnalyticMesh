""" A simple chair model
"""
import os
from AnalyticMesh import load_model, AnalyticMarching

if __name__ == "__main__":
    DIR = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(DIR, "chair.onnx")
    model = load_model(onnx_path)

    ply_path = os.path.join(DIR, "chair.ply")
    AnalyticMarching(model, ply_path)
    print(f"we save ply to: {ply_path}")
