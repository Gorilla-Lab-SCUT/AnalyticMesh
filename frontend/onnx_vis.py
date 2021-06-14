
import json
import pydot
from onnx import ModelProto
from collections import defaultdict


OP_STYLE = {
    'shape': 'box',
    'color': '#003f10',
    'style': 'filled',
    'fontcolor': '#FFFFFF',
    'fontsize': 20,
    'fontname': 'times bold',
}

BLOB_STYLE = {
    'shape': 'box', 
    'style': 'rounded',
    'fontsize': 20,
    'fontname': 'times bold',
}


class OnnxToPng(object):
    def __init__(self):
        return

    def _escape_label(self, name):
        return json.dumps(name)

    def GetOpNodeProducer(self, **kwargs):
        def ReallyGetOpNode(op, op_id):
            if op.name:
                node_name = '%s/%s (op#%d)' % (op.name, op.op_type, op_id)
            else:
                node_name = '%s (op#%d)' % (op.op_type, op_id)
            node = pydot.Node(node_name, **kwargs)
            return node

        return ReallyGetOpNode

    def GetPydotGraph(
            self,
            graph,
            name=None,
            rankdir='LR',
            node_producer=None,
    ):
        if node_producer is None:
            node_producer = self.GetOpNodeProducer(**OP_STYLE)
        pydot_graph = pydot.Dot(name, rankdir=rankdir)
        pydot_nodes = {}
        pydot_node_counts = defaultdict(int)
        for op_id, op in enumerate(graph.node):
            op_node = node_producer(op, op_id)
            pydot_graph.add_node(op_node)
            for input_name in op.input:
                if input_name not in pydot_nodes:
                    input_node = pydot.Node(
                        self._escape_label(
                            input_name + str(pydot_node_counts[input_name])),
                        label=self._escape_label(input_name),
                        **BLOB_STYLE
                    )
                    pydot_nodes[input_name] = input_node
                else:
                    input_node = pydot_nodes[input_name]
                pydot_graph.add_node(input_node)
                pydot_graph.add_edge(pydot.Edge(input_node, op_node))
            for output_name in op.output:
                if output_name in pydot_nodes:
                    pydot_node_counts[output_name] += 1
                output_node = pydot.Node(
                    self._escape_label(
                        output_name + str(pydot_node_counts[output_name])),
                    label=self._escape_label(output_name),
                    **BLOB_STYLE
                )
                pydot_nodes[output_name] = output_node
                pydot_graph.add_node(output_node)
                pydot_graph.add_edge(pydot.Edge(op_node, output_node))
        return pydot_graph

    def run(self, onnx_model):
        model = ModelProto()
        content = onnx_model
        model.ParseFromString(content)

        pydot_graph = self.GetPydotGraph(
            model.graph,
            name=model.graph.name,
            rankdir='TD',
            node_producer=self.GetOpNodeProducer(
                **OP_STYLE
            ),
        )
        return pydot_graph.create(format='png')


def onnx_vis(onnx_path):
    """ read an onnx file, and return png file in bytes
    """
    convertor = OnnxToPng()
    with open(onnx_path, "rb") as f:
        onnx_model = f.read()
    png_file_bytes = convertor.run(onnx_model)
    return png_file_bytes
