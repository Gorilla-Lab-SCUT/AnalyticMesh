""" pytest -s -p no:warnings test_onnx_io.py
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import torch
import inspect
import numpy as np
from backend import AnalyticMarching
from backend import MLP
from backend import save_model, load_model

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tmp")
os.makedirs(DIR, exist_ok=True)

class RedirectStream:
    def __init__(self, redirect_file_path="/dev/null"):
        self.f = open(redirect_file_path, 'w')
        self.devnull = open(os.devnull, 'w')
        self._stdout = self.f or self.devnull or sys.stdout
        self._stderr = self.f or self.devnull or sys.stderr

    def start(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def close(self):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()
        self.f.close()

    def flush(self):
        self._stdout.flush()
        self._stderr.flush()

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def run(mlp, onnx_path="model.onnx", ply_path="model.ply"):

    save_model(mlp, onnx_path)
    print(f"we save onnx to: {onnx_path}")

    nn_model = load_model(onnx_path)
    print(f"nn_model nodes = {nn_model.nodes}")
    print(f"nn_model arc_table = {nn_model.arc_table}")
    print(f"nn_model arc_tm_shape = {nn_model.arc_tm_shape}")

    with RedirectStream():
        AnalyticMarching(model=nn_model, save_ply_path=ply_path)
    print(f"we save mesh to: {ply_path}")


def test_custom_1():
    class MyMLP(torch.nn.Module):
        def __init__(self):
            super(MyMLP, self).__init__()
            self.nodes = [3, 100, 100, 100, 1]
            self.geometric_radius = 0.5
            self.linear0 = torch.nn.Linear(3, self.nodes[1])
            self.linear1 = torch.nn.Linear(self.nodes[1], self.nodes[2])
            self.linear2 = torch.nn.Linear(self.nodes[2], self.nodes[3])
            self.linear3 = torch.nn.Linear(self.nodes[3], 1)

            def init_layer(linear, not_last=True):
                if not_last:
                    torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(self.nodes[i + 1]))
                    if linear.bias is not None:
                        torch.nn.init.zeros_(linear.bias)
                else:
                    torch.nn.init.constant_(linear.weight, np.sqrt(np.pi) / np.sqrt(self.nodes[i]))
                    if linear.bias is not None:
                        torch.nn.init.constant_(linear.bias, -self.geometric_radius)
            for i in range(4):
                linear = getattr(self, f"linear{i}")
                init_layer(linear, not_last=(i!=3))

        def forward(self, x):
            x1 = x
            y1 = torch.relu(self.linear0(x1))
            y2 = torch.relu(self.linear1(y1))
            y3 = torch.relu(self.linear2(y2))
            return self.linear3(y3)

    mlp = MyMLP()
    prefix = inspect.getframeinfo(inspect.currentframe())[2]
    run(mlp, onnx_path=os.path.join(DIR, f"{prefix}.onnx"), ply_path=os.path.join(DIR, f"{prefix}.ply"))


def test_custom_2():
    class MyMLP(torch.nn.Module):
        def __init__(self):
            super(MyMLP, self).__init__()
            self.nodes = [3, 20, 50, 20, 80, 20, 1]
            self.geometric_radius = 0.5
            self.linear0 = torch.nn.Linear(3, self.nodes[1])
            self.linear1 = torch.nn.Linear(self.nodes[1], self.nodes[2])
            self.linear2 = torch.nn.Linear(self.nodes[2], self.nodes[3])
            self.linear3 = torch.nn.Linear(self.nodes[3], self.nodes[4])
            self.linear4 = torch.nn.Linear(self.nodes[4], self.nodes[5])
            self.linear5 = torch.nn.Linear(self.nodes[5], 1)
            self.skip0 = torch.nn.Linear(self.nodes[1], self.nodes[4], bias=False)
            self.skip1 = torch.nn.Linear(self.nodes[2], 1, bias=False)

            def init_layer(linear, not_last=True):
                if not_last:
                    torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(self.nodes[i + 1]))
                    if linear.bias is not None:
                        torch.nn.init.zeros_(linear.bias)
                else:
                    torch.nn.init.constant_(linear.weight, np.sqrt(np.pi) / np.sqrt(self.nodes[i]))
                    if linear.bias is not None:
                        torch.nn.init.constant_(linear.bias, -self.geometric_radius)
            for i in range(6):
                linear = getattr(self, f"linear{i}")
                init_layer(linear, not_last=(i!=5))
            for i in range(2):
                linear = getattr(self, f"skip{i}")
                init_layer(linear, not_last=(i!=1))

        def forward(self, x):
            x1 = x
            y1 = torch.relu(self.linear0(x1))
            y2 = torch.relu(self.linear1(y1))
            y3 = torch.relu(self.linear2(y2) + y1)
            y4 = torch.relu(self.linear3(y3) + self.skip0(y1))
            y5 = torch.relu(self.linear4(y4))
            return self.linear5(y5) + self.skip1(y2)

    mlp = MyMLP()
    prefix = inspect.getframeinfo(inspect.currentframe())[2]
    run(mlp, onnx_path=os.path.join(DIR, f"{prefix}.onnx"), ply_path=os.path.join(DIR, f"{prefix}.ply"))

def test_mlp_1():
    mlp = MLP(nodes=[3, 100, 100, 1], arc_table=[[0], [0]], arc_tm_shape=[], geometric_radius=0.5)
    prefix = inspect.getframeinfo(inspect.currentframe())[2]
    run(mlp, onnx_path=os.path.join(DIR, f"{prefix}.onnx"), ply_path=os.path.join(DIR, f"{prefix}.ply"))

def test_mlp_2():
    mlp = MLP(nodes=[3, 80, 80, 80, 80, 80, 1],
              arc_table=[[1, 0, 0], [1, 1, 1], [0], [1, 3, 2], [1, 0, 3]],
              arc_tm_shape=[[80, 3], [0, 0], [80, 80], [1, 3]],
              geometric_radius=0.5)
    prefix = inspect.getframeinfo(inspect.currentframe())[2]
    run(mlp, onnx_path=os.path.join(DIR, f"{prefix}.onnx"), ply_path=os.path.join(DIR, f"{prefix}.ply"))
