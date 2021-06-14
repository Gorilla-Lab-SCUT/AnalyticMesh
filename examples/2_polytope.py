""" A simple polytope
    It gives an example of how to apply analytic marching on your custom network
"""
import os
import torch
from AnalyticMesh import save_model, load_model, AnalyticMarching

class MLPPolytope(torch.nn.Module):
    def __init__(self):
        super(MLPPolytope, self).__init__()
        self.linear0 = torch.nn.Linear(3, 14)
        self.linear1 = torch.nn.Linear(14, 1)
        with torch.no_grad():
            weight0 = torch.tensor([[ 1,  1,  1],
                                    [-1, -1, -1],
                                    [ 0,  1,  1],
                                    [ 0, -1, -1],
                                    [ 1,  0,  1],
                                    [-1,  0, -1],
                                    [ 1,  1,  0],
                                    [-1, -1,  0],
                                    [ 1,  0,  0],
                                    [-1,  0,  0],
                                    [ 0,  1,  0],
                                    [ 0, -1,  0],
                                    [ 0,  0,  1],
                                    [ 0,  0, -1]], dtype=torch.float32)
            bias0 = torch.zeros(14)
            weight1 = torch.ones([14], dtype=torch.float32).unsqueeze(0)
            bias1 = torch.tensor([-2], dtype=torch.float32)

            add_noise = lambda x: x + torch.randn_like(x) * (1e-7)
            self.linear0.weight.copy_(add_noise(weight0))
            self.linear0.bias.copy_(add_noise(bias0))
            self.linear1.weight.copy_(add_noise(weight1))
            self.linear1.bias.copy_(add_noise(bias1))

    def forward(self, x):
        return self.linear1(torch.relu(self.linear0(x)))


if __name__ == "__main__":
    #### save onnx
    DIR = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(DIR, "polytope.onnx")
    save_model(MLPPolytope(), onnx_path)
    print(f"we save onnx to: {onnx_path}")

    #### save ply
    ply_path = os.path.join(DIR, "polytope.ply")
    model = load_model(onnx_path) # load as a specific model
    AnalyticMarching(model, ply_path)
    print(f"we save ply to: {ply_path}")
