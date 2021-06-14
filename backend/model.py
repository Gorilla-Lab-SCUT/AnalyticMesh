import torch
import numpy as np


class MLP(torch.nn.Module):
    """ MLP of arbitrary architecture (specialized for cuam library)
    """
    def __init__(
        self,
        nodes,
        arc_table,
        arc_tm_shape,
        initialization='geometric',  # `kaiming` or (`geometric` as in SAL paper)
        geometric_radius=1.0,
        enable_print=True,
    ):
        """initialization for MLP

        Args:
            nodes (list): The num of nodes of each layer (including input layer and output layer).
                For example, nodes = [3, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1].

            arc_table (list): A list of list which specifies the architecture of MLP. 
                For example, arc_table = [[0], [1, 1, 0], [0], [1, 3, 0], [0], [1, 5, 0], [0], [1, 7, 0], [0], [0]].
                Note: 
                    len(arc_table) should be equal to the num of hidden layers, i.e. len(nodes)-2
                    arc_table[i][0] should be the number of incoming adding skip connection (0 <= i < len(arc_table))
                    len(arc_table[i]) should be of length 1+2*arc_table[i][0]
                    arc_table[i][1+2*j] should be the index of the source of the adding skip connection (0 <= j < arc_table[i][0])
                    arc_table[i][2+2*j] should be the index of the transformation matrix (0 <= j < arc_table[i][0])

            arc_tm_shape (list): list of the shape of transformation matrices. 
                For example, arc_tm_shape = [[0, 0]].
                Note:
                    if it is an identity matrix, please use shape==[0, 0] for performance optimization
                    
            initialization (str, optional): choose the method to initialize the parameters. options are 'kaiming' or 'geometric'.
                Defaults to 'geometric'.
                Note:
                    'kaiming' is the well-known kaiming initialization strategy
                    'geometric' is Geometric network initialization strategy described in 
                        paper `SAL: Sign Agnostic Learning of Shapes from Raw Data`. 
                        It is suitable for fitting a Signed Distance Field from raw point cloud (without normals or any sign).
            
            geometric_radius (float, optional): the radius of the ball when geometric initialization is used
                Defaults to 1.0
        """
        super(MLP, self).__init__()
        self.nodes = nodes
        self.arc_table = arc_table
        self.arc_tm_shape = arc_tm_shape
        self.geometric_radius = geometric_radius

        # linear
        self.num_of_linears = len(self.nodes) - 1
        self.linears = torch.nn.ModuleList()
        for i in range(self.num_of_linears):
            linear = torch.nn.Linear(self.nodes[i], self.nodes[i + 1], bias=True)
            if initialization is not None:
                if initialization == 'kaiming':
                    torch.nn.init.kaiming_normal_(linear.weight)
                    torch.nn.init.zeros_(linear.bias)
                elif initialization == 'geometric':
                    # from SAL paper
                    if i != self.num_of_linears - 1:
                        torch.nn.init.normal_(linear.weight, 0.0, np.sqrt(2) / np.sqrt(self.nodes[i + 1]))
                        torch.nn.init.zeros_(linear.bias)
                    else:
                        torch.nn.init.constant_(linear.weight, np.sqrt(np.pi) / np.sqrt(self.nodes[i]))
                        torch.nn.init.constant_(linear.bias, -self.geometric_radius)
                else:
                    raise Exception(f'Error: No such initialization: {initialization}')
            elif i == 0:
                if enable_print:
                    print(f"Warning: Initialization strategy for MLP is not specified.")
            self.linears.append(linear)

        # activation
        self.num_of_acts = self.num_of_linears
        self.acts = torch.nn.ModuleList()
        for i in range(self.num_of_acts):
            self.acts.append(torch.nn.ReLU(inplace=True) if i != self.num_of_acts - 1 else torch.nn.Identity())

        # transform matrix
        self.num_of_tms = len(self.arc_tm_shape)
        self.tms = torch.nn.ModuleList()
        for tm_shape in self.arc_tm_shape:
            if tm_shape[0] == 0 and tm_shape[1] == 0:
                self.tms.append(torch.nn.Identity())
            else:
                self.tms.append(torch.nn.Linear(in_features=tm_shape[1], out_features=tm_shape[0], bias=False))

        # source of transform
        self.srcs = {}
        for arc in self.arc_table:
            for i in range(arc[0]):
                self.srcs[arc[1 + 2 * i]] = None

        self.outputs_list = []

    def forward(self, x, requires_outputs_list=False):
        if requires_outputs_list:
            self.outputs_list.clear()

        for i in range(self.num_of_linears):
            if i in self.srcs.keys():
                self.srcs[i] = x

            x = self.linears[i](x)

            if i >= 1:
                for j in range(self.arc_table[i - 1][0]):
                    src = self.srcs[self.arc_table[i - 1][1 + 2 * j]]
                    transform = self.tms[self.arc_table[i - 1][2 + 2 * j]]
                    x = x + transform(src)

            x = self.acts[i](x)

            if i != self.num_of_linears - 1:
                if requires_outputs_list:
                    self.outputs_list.append(x)

        return x

    def get_info(self):

        weights = []
        for i in range(self.num_of_linears):
            weights.append(self.linears[i].weight)

        biases = []
        for i in range(self.num_of_linears):
            biases.append(self.linears[i].bias)

        arc_tm = []
        for i in range(self.num_of_tms):
            if isinstance(self.tms[i], torch.nn.Identity):
                arc_tm.append(torch.zeros([0, 0]))
            elif isinstance(self.tms[i], torch.nn.Linear):
                arc_tm.append(self.tms[i].weight)

        arc_table_width = max([len(arc) for arc in self.arc_table])
        arc_table = torch.zeros([len(self.arc_table), arc_table_width], dtype=torch.int32)
        for r in range(len(self.arc_table)):
            for c in range(len(self.arc_table[r])):
                arc_table[r, c] = self.arc_table[r][c]

        return {'weights': weights, 'biases': biases, 'arc_tm': arc_tm, 'arc_table': arc_table}
