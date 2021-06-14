""" A simple sphere
    Increasing the width will make it more spherical
"""
import os
from AnalyticMesh import MLP, save_model, AnalyticMarching

if __name__ == "__main__":
    NODES = [3, 500, 500, 1]
    model = MLP(nodes=NODES, arc_table=[[0]] * (len(NODES) - 2), arc_tm_shape=[], geometric_radius=0.5)

    #### save onnx
    DIR = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(DIR, "sphere.onnx")
    save_model(model, onnx_path)
    print(f"we save onnx to: {onnx_path}")

    #### save ply
    ply_path = os.path.join(DIR, "sphere.ply")
    AnalyticMarching(model, ply_path)
    print(f"we save ply to: {ply_path}")
