""" parse onnx model and load as torch.nn.Module
"""
import os
import re
import onnx
import torch
from .model import MLP
from onnx.numpy_helper import to_array 
from io import BytesIO

def _onnx2mlp(model_graph):
    
    def find(dict_list, name, key="name", fn=lambda name, value: value==name):
        """ helper function
        """
        for d in dict_list:
            if fn(name, d[key]):
                return d

    ###################################################################
    def parse_str():
        """ parse necessary info from graph_str
        """
        pattern = re.compile(r"graph\s(?P<name>\b[^\s]+\b)\s"
                             r"\((?P<input_val>.+?)\) initializers\s"
                             r"\((?P<params>.+?)\)\s"
                             r"\{(?P<flows>.+?)\}"
                             , re.DOTALL)
        groupdict = pattern.match(graph_str).groupdict()
        name = groupdict["name"]

        pattern = re.compile(f"%[^\n]+(?=\n)", re.DOTALL)
        input_val = pattern.findall(groupdict["input_val"])

        pattern = re.compile(f"%[^\n]+(?=\n)", re.DOTALL)
        params = pattern.findall(groupdict["params"])

        pattern = re.compile(f"(?<!return\s)%[^\n]+(?=\n)", re.DOTALL)
        flows = pattern.findall(groupdict["flows"])

        pattern = re.compile(f"(?<=return\s)%[^\n]+(?=\n)", re.DOTALL)
        return_val = pattern.findall(groupdict["flows"])

        assert len(input_val) == 1
        input_val = input_val[0]
        assert len(return_val) == 1
        return_val = return_val[0]

        def parse_tensor(tensor_str):
            tensor = re.match(r"%(?P<name>.*)\[(?P<dtype>\w+),\s(?P<shape>\w+)\]", tensor_str).groupdict()
            tensor = dict(name=tensor["name"], 
                          dtype=tensor["dtype"],
                          shape=tuple([int(s) for s in re.split(r"x", tensor["shape"])]))
            return tensor

        input_val = parse_tensor(input_val)
        params = [parse_tensor(s) for s in params]

        flows_ = list()
        for flow in flows:
            groupdict = re.match(r"(?P<out>[^=\s]+)\s=\s"
                                 r"(?P<fun_name>\w+)"
                                 r"(\[(?P<fun_configs>.+)\])?"
                                 r"\((?P<fun_args>.+)\)", flow).groupdict()

            if groupdict["out"]:
                groupdict["out"] = groupdict["out"].replace("%", "")

            if groupdict["fun_configs"]:
                groupdict["fun_configs"] = eval("dict(" + groupdict["fun_configs"] + ")")

            if groupdict["fun_args"]:
                groupdict["fun_args"] = tuple([s.replace("%", "") for s in re.split(r"(?:,\s)+", groupdict["fun_args"])])
            
            flows_.append(groupdict)

        flows = flows_

        return_val = return_val.replace("%", "")

        graph = dict(name=name, input_val=input_val, params=params, flows=flows, return_val=return_val)
        return graph

    ###################################################################
    def check_graph():
        """ assert it is valid mlp
        """
        # check input_val
        assert graph["input_val"]["shape"][1] == 3 # in dim == 3
        assert graph["input_val"]["dtype"] == "FLOAT"

        # check params
        for p in graph["params"]:
            assert p["dtype"] == "FLOAT"
        
        # check flows
        do_transpose = False
        expect_next = None
        for f in graph["flows"]:
            assert f["fun_name"] in ["Gemm", "Relu", "MatMul", "Add", "Transpose"], f"Found unrecognized function: {f['fun_name']}\ngraph_str: {graph_str}"
            if expect_next:
                assert f["fun_name"] == expect_next, "Transpose must follow by MatMul"
                expect_next = None
            if f["fun_configs"]:
                if f["fun_name"] == "Gemm":
                    assert f["fun_configs"]["alpha"] == 1
                    assert f["fun_configs"]["beta"] == 1
                    assert f["fun_configs"]["transB"] == 1
                elif f["fun_name"] == "Transpose":
                    assert f["fun_configs"]["perm"] == [1, 0]
                    do_transpose = True
                    expect_next = "MatMul"

        if do_transpose: # remove Transpose
            for idx in range(len(graph["flows"])):
                f = graph["flows"][idx]
                if f["fun_name"] == "MatMul":
                    pre_f = graph["flows"][idx - 1]
                    assert pre_f["fun_name"] == "Transpose"
                    f["fun_args"] = tuple([f["fun_args"][0], pre_f["fun_args"][0]])
                    graph["flows"][idx - 1] = None # remove

            graph["flows"] = [f for f in graph["flows"] if f is not None]

        return do_transpose

        # check return_val
        gemm_list = [f for f in graph["flows"] if f["fun_name"] == "Gemm"]
        assert find(graph["params"], gemm_list[-1]["fun_args"][1])["shape"][0] == 1

    ###################################################################
    def infer_relu_tensors():
        """ infer intermediate relu tensors' info (including shapes)
        """
        gemm_list = [f for f in graph["flows"] if f["fun_name"] == "Gemm"]
        relu_list = [f for f in graph["flows"] if f["fun_name"] == "Relu"]
        assert len(relu_list) == len(gemm_list) - 1
        relu_tensors = [None for _ in range(len(relu_list))]
        for idx, (f_gemm, f_relu) in enumerate(zip(gemm_list[:-1], relu_list)):
                name = f_relu["out"]
                dtype = "FLOAT"
                batch = graph["input_val"]["shape"][0]
                out_dim = find(graph["params"], f_gemm["fun_args"][1])["shape"][0]
                shape = (batch, out_dim)
                relu_tensors[idx] = dict(name=name, dtype=dtype, shape=shape)
        return relu_tensors

    ###################################################################
    def get_nodes():
        nodes_len = len(relu_tensors) + 2
        nodes = [None for _ in range(nodes_len)]
        for node_i in range(nodes_len):
            if node_i == 0:
                nodes[node_i] = 3
            elif node_i == nodes_len - 1:
                nodes[node_i] = 1
            else:
                nodes[node_i] = relu_tensors[node_i - 1]["shape"][1]
        
        return nodes

    ###################################################################
    def get_arch():
        relu_list = [f for f in graph["flows"] if f["fun_name"] == "Relu"]
        aux_nodes = [None for _ in range(len(nodes))]
        for i in range(len(aux_nodes)):
            if i == 0:
                aux_nodes[i] = graph["input_val"]["name"]
            elif i == len(aux_nodes) - 1:
                aux_nodes[i] = graph["return_val"]
            else:
                aux_nodes[i] = relu_list[i - 1]["out"]
        
        arc_table = list()
        arc_tm_shape = list()
        shortcuts = list()
        for f in graph["flows"]:
            if f["fun_name"] == "Add":

                # determine dst index
                dst_str = f["out"]
                f2 = None
                while (dst_str != graph["return_val"]) and ((f2 is None) or (f2["fun_name"]!="Relu")):
                    f2 = find(graph["flows"], dst_str, "fun_args", lambda name, value: (name in value))
                    dst_str = f2["out"]
                dst_index = aux_nodes.index(dst_str)

                # determine src index
                for add_arg in f["fun_args"]:
                    f2 = find(graph["flows"], add_arg, "out")
                    if f2["fun_name"] in ["Gemm", "Add"]:
                        continue
                    elif f2["fun_name"] == "Relu":
                        src_index = aux_nodes.index(add_arg)
                        tm_name = "identity_matrix"
                        break
                    elif f2["fun_name"] == "MatMul":
                        src_index = aux_nodes.index(f2["fun_args"][0])
                        tm_name = f2["fun_args"][1]
                        break
                
                shortcuts.append(dict(src_index=src_index, dst_index=dst_index, tm_name=tm_name))
        
        arc_table = [[0] for _ in range(len(nodes) - 2)]
        arc_tm_shape = list()
        arc_tm_shape_name = list()
        for short_cut in shortcuts:
            if short_cut["tm_name"] not in arc_tm_shape_name:
                arc_tm_shape_name.append(short_cut["tm_name"])
                if short_cut["tm_name"] != "identity_matrix":
                    fn_swap = lambda shape2d: shape2d if do_transpose else [shape2d[1], shape2d[0]]
                    arc_tm_shape.append(fn_swap(find(graph["params"], short_cut["tm_name"], key="name")["shape"]))
                else:
                    arc_tm_shape.append([0, 0])

            arc_table[short_cut["dst_index"] - 2][0] += 1
            arc_table[short_cut["dst_index"] - 2].extend([short_cut["src_index"], arc_tm_shape_name.index(short_cut["tm_name"])])
        
        return arc_table, arc_tm_shape, arc_tm_shape_name

    ###################################################################
    def get_info():
        def load_param(name):
            [tensor] = [t for t in model_graph.initializer if t.name == name]
            param = torch.from_numpy(to_array(tensor).copy())
            return param
        
        transpose = lambda tensor: tensor if do_transpose else tensor.t()
        weights = [load_param(f["fun_args"][1]) for f in graph["flows"] if f["fun_name"] == "Gemm"]
        biases = [load_param(f["fun_args"][2]) for f in graph["flows"] if f["fun_name"] == "Gemm"]
        arc_tm = [(transpose(load_param(p)) if p != "identity_matrix" else torch.zeros([0, 0])) for p in arc_tm_shape_name]

        return dict(weights=weights, biases=biases, arc_tm=arc_tm)
    
    ###################################################################
    def get_nn_model():
        nn_model = MLP(nodes=nodes, arc_table=arc_table, arc_tm_shape=arc_tm_shape, initialization=None, enable_print=False)
        with torch.no_grad():
            for i in range(nn_model.num_of_linears):
                nn_model.linears[i].weight.copy_(info["weights"][i])
                nn_model.linears[i].bias.copy_(info["biases"][i])

            for i in range(nn_model.num_of_tms):
                if isinstance(nn_model.tms[i], torch.nn.Linear):
                    nn_model.tms[i].weight.copy_(info["arc_tm"][i])

        return nn_model

    ###################################################################
    graph_str = onnx.helper.printable_graph(model_graph) # to readable str
    graph = parse_str()
    do_transpose = check_graph()
    relu_tensors = infer_relu_tensors()
    nodes = get_nodes()
    arc_table, arc_tm_shape, arc_tm_shape_name = get_arch()
    info = get_info()
    nn_model = get_nn_model()

    return nn_model


def save_model(model, model_path):
    dummy_input = torch.randn([1, 3])
    model.cpu().float()
    torch.onnx.export(model, dummy_input, model_path)


def load_model(model_file_str):
    """ load a onnx model

    Args:
        model_file_str (str or bytes): if str, it gives the path to onnx model file; 
                                       if bytes, it gives the binary content of the onnx model file
    Returns:
        nn_model (torch.nn.Module): A torch.nn.Module model
    """
    if isinstance(model_file_str, str) and os.path.exists(model_file_str):
        model = onnx.load(model_file_str)
    elif isinstance(model_file_str, bytes):
        model_file = BytesIO(model_file_str)
        model = onnx.load(model_file)
    else:
        raise Exception(f"Error: unknown model_file_str = {model_file_str}")
    nn_model = _onnx2mlp(model.graph)
    return nn_model

