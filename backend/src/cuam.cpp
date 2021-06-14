#include <torch/extension.h>
#include <ctime>
#include <vector>
#include <string>

namespace py = pybind11;

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!(x.is_cuda()), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)       \
    {                        \
        CHECK_CUDA(x);       \
        CHECK_CONTIGUOUS(x); \
    }
#define CHECK_HOST(x)        \
    {                        \
        CHECK_CPU(x);        \
        CHECK_CONTIGUOUS(x); \
    }

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

void Init_cuda(const std::string &float_type,
               const std::vector<int> &nodesnum,
               const torch::Tensor &arc_table,
               int num_extra_constraints);

void AnalyticMarching_cuda(const std::vector<torch::Tensor> &weights,
                           const std::vector<torch::Tensor> &biases,
                           const torch::Tensor &states,
                           const torch::Tensor &points,
                           const std::vector<torch::Tensor> &arc_tm,
                           const torch::Tensor &w_extra_constraints,
                           const torch::Tensor &b_extra_constraints,
                           double iso,
                           bool flip_insideout);

void CombineMesh(double scale, std::vector<double> center);

void ExportMesh(std::string file_path,
                bool is_polymesh,
                bool is_float32);

void Destroy();

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

std::string float_type_;
std::vector<int> nodesnum_;
torch::Tensor arc_table_;

void Init(std::string float_type,
          std::vector<int> nodesnum,
          torch::Tensor arc_table,   // CPU, int32
          int num_extra_constraints) // >= 0
{
    if (float_type != "float32" && float_type != "float64")
    {
        std::cout << "Error: `float_type` is either `float32` or `float64`!";
        return;
    }

    // check ==> nodesnum
    TORCH_CHECK(nodesnum.front() == 3);
    TORCH_CHECK(nodesnum.back() == 1);
    TORCH_CHECK(nodesnum.size() >= 3);

    // check ==> arc_table
    CHECK_HOST(arc_table);
    TORCH_CHECK(arc_table.dim() == 2);
    TORCH_CHECK(arc_table.size(0) == nodesnum.size() - 2);
    TORCH_CHECK(arc_table.size(1) >= 1 && (arc_table.size(1) + 1) % 2 == 0);
    TORCH_CHECK(arc_table.dtype() == torch::kInt32);
    auto arc_table_acc = arc_table.accessor<int, 2>();
    for (int i = 0; i < int(arc_table.size(0)); ++i)
    {
        int connection_num = arc_table_acc[i][0];
        TORCH_CHECK(1 + 2 * connection_num <= arc_table.size(1));
    }

    float_type_ = float_type;
    nodesnum_ = nodesnum;
    arc_table_ = arc_table;

    TORCH_CHECK(num_extra_constraints >= 0);

    // run
    Init_cuda(float_type, nodesnum, arc_table, num_extra_constraints);
}

void AnalyticMarching(std::vector<torch::Tensor> weights,       // weights of MLP, shape==(out, in), FloatType, GPU
                      std::vector<torch::Tensor> biases,        // biases of MLP, shape==(out, ), FloatType, GPU
                      torch::Tensor states,                     // 0-1 binary mask, row samples, shape==(num_of_init_points, hidden_states_vector_len), bool, GPU
                      torch::Tensor points,                     // points of states, shape==(num_of_init_points, 3), FloatType, GPU
                      std::vector<torch::Tensor> arc_tm,        // transform matrices, dim==2, FloatType, GPU
                      const torch::Tensor &w_extra_constraints, // extra weights of constraints (<0), shape==(num_extra_constraints, 3), FloatType, GPU
                      const torch::Tensor &b_extra_constraints, // extra biases of constraints (<0), shape==(num_extra_constraints, ), FloatType, GPU
                      double iso,
                      bool flip_insideout)
{
    if (float_type_ == "")
    {
        std::cout << "Environment must be initialized first!" << std::endl;
        return;
    }
    auto TorchFloatType = (float_type_ == "float32" ? torch::kFloat32 : torch::kFloat64);

    // check ==> weights && biases
    TORCH_CHECK(weights.size() == biases.size());
    const int fc_layers_num = int(weights.size());
    TORCH_CHECK(fc_layers_num == nodesnum_.size() - 1);
    int hidden_states_vector_len = 0;
    for (int i = 0; i < fc_layers_num; ++i)
    {
        CHECK_INPUT(weights[i]);
        CHECK_INPUT(biases[i]);
        TORCH_CHECK(weights[i].dim() == 2);
        TORCH_CHECK(biases[i].dim() == 1);
        TORCH_CHECK(weights[i].size(0) == biases[i].size(0));
        if (i != 0)
        {
            TORCH_CHECK(weights[i].size(1) == weights[i - 1].size(0));
        }
        hidden_states_vector_len += weights[i].size(0);
        TORCH_CHECK(weights[i].dtype() == biases[i].dtype());
        TORCH_CHECK(weights[i].dtype() == TorchFloatType);
        TORCH_CHECK(nodesnum_[i + 1] == weights[i].size(0));
    }
    TORCH_CHECK(weights[0].size(1) == 3);
    TORCH_CHECK(weights.back().size(0) == 1);
    hidden_states_vector_len -= 1;

    // check ==> states
    CHECK_INPUT(states);
    TORCH_CHECK(states.dim() == 2);
    TORCH_CHECK(states.size(0) >= 1);
    TORCH_CHECK(states.size(1) == hidden_states_vector_len);
    TORCH_CHECK(states.dtype() == torch::kBool);

    // check ==> points
    CHECK_INPUT(points);
    TORCH_CHECK(points.dim() == 2);
    TORCH_CHECK(points.size(0) == states.size(0));
    TORCH_CHECK(points.size(1) == 3);
    TORCH_CHECK(points.dtype() == TorchFloatType);

    // check ==> arc_tm
    for (auto &tm : arc_tm)
    {
        CHECK_INPUT(tm);
        TORCH_CHECK(tm.dim() == 2);
        TORCH_CHECK(tm.dtype() == TorchFloatType);
    }
    auto arc_table_acc = arc_table_.accessor<int, 2>();
    for (int i = 0; i < fc_layers_num - 1; ++i)
    {
        for (int j = 0; j < arc_table_acc[i][0]; ++j)
        {
            int from = arc_table_acc[i][j * 2 + 1];
            int idx = arc_table_acc[i][j * 2 + 2]; // idx to transform matrix
            if (arc_tm[idx].size(0) || arc_tm[idx].size(1))
            {
                TORCH_CHECK(arc_tm[idx].size(0) == weights[i + 1].size(0));
                TORCH_CHECK(arc_tm[idx].size(1) == weights[from].size(1));
            }
        }
    }

    CHECK_INPUT(w_extra_constraints);
    CHECK_INPUT(b_extra_constraints);
    TORCH_CHECK(w_extra_constraints.dim() == 2);
    TORCH_CHECK(w_extra_constraints.size(1) == 3);
    TORCH_CHECK(b_extra_constraints.dim() == 1);
    TORCH_CHECK(w_extra_constraints.size(0) == b_extra_constraints.size(0));

    // run
    AnalyticMarching_cuda(weights, biases, states, points, arc_tm, w_extra_constraints, b_extra_constraints, iso, flip_insideout);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "A CUDA-based implementation of Analytic Marching algorithm";

    m.def("Init", &Init, "Initialize environment (CUDA)",
          py::arg("float_type"),
          py::arg("nodesnum"),
          py::arg("arc_table"),
          py::arg("num_extra_constraints"));

    m.def("AnalyticMarching", &AnalyticMarching, "AnalyticMarching (CUDA)",
          py::arg("weights"),
          py::arg("biases"),
          py::arg("states"),
          py::arg("points"),
          py::arg("arc_tm"),
          py::arg("w_extra_constraints"),
          py::arg("b_extra_constraints"),
          py::arg("iso"),
          py::arg("flip_insideout"));

    m.def("CombineMesh", &CombineMesh, "Combine to a mesh (CUDA)",
          py::arg("scale"),
          py::arg("center"));

    m.def("ExportMesh", &ExportMesh, "Export mesh file (CUDA)",
          py::arg("file_path"),
          py::arg("is_polymesh"),
          py::arg("is_float32"));

    m.def("Destroy", &Destroy, "Destroy environment (CUDA)");
}