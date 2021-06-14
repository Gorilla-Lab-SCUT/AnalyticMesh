
#include <torch/extension.h>
#include <iostream>
#include <ctime>
#include <vector>
#include <string>

#include "utilities.h"
#include "facedata.h" 
#include "states.h" 
#include "mlp.h" 
#include "polymesh.h"
#include "kernel.h" 
#include "process.h" 
#include "var.h"

#include "vecmat.h"

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

std::string FloatTypeString = ""; 
template <typename FloatType> VAR<FloatType>* var_ptr = nullptr;
bool has_am = false;
bool has_calc = false;

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

template <typename FloatType>
void main_(const std::vector<torch::Tensor> &weights_,
           const std::vector<torch::Tensor> &biases_,
           const torch::Tensor &states_,
           const torch::Tensor &points_,
           const std::vector<torch::Tensor> &arc_tm_,
           const torch::Tensor &w_extra_constraints_,
           const torch::Tensor &b_extra_constraints_,
           FloatType iso,
           bool flip_insideout,
           VAR<FloatType>& var)
{
    int batch_size;
    FloatType* states_buffer;
    FACEDATA<FloatType>* faces_buffer;
    
    MAT<FloatType> w_extra_constraints(w_extra_constraints_);
    VEC<FloatType> b_extra_constraints(b_extra_constraints_);

    var.mlp.reinit(weights_, biases_, arc_tm_);
    var.states.reinit(states_, points_); // it can accept duplicated states (it will remove them)
    var.polymesh.reinit();
    
    while(var.states.states_num > 0)
    {
        batch_size = MIN2(var.states.states_num, var.BATCH_SIZE_MAX);
        var.states.pop_back(states_buffer, faces_buffer, batch_size);

        // get constraints (output: w_inequ, b_inequ, w_equ, b_equ)
        fill_constraints(var.mlp, var.states, batch_size, 
                         var.w_inequ, var.b_inequ, var.w_equ, var.b_equ, var.w_temp, var.b_temp, 
                         w_extra_constraints, b_extra_constraints, var.num_extra_constraints,
                         states_buffer, var.handle); // 
        
        // add iso offset to `b_equ`
        offset_b_equ(var.b_equ, iso, batch_size);

        // calc distance (output: distance)
        fill_distance(var.states, var.w_inequ, var.b_inequ, var.w_equ, batch_size, 
                      faces_buffer, var.distance, var.num_extra_constraints, var.handle);
        
        // calc priority in descending order (output: dist_idx)
        sortDistance(var.distance, var.segments, var.key_vec, 
                     var.states.state_len + var.num_extra_constraints, 
                     batch_size);

        // fill start_idx for pivoting (output: faces_buffer)
        //   if the value is still -1, then it must not be processed.
        fillStartIdx(faces_buffer, var.dist_idx, var.As, var.bs, var.xs, var.cnt_x, var.SOLVE_SIZE_MAX,
                     var.w_inequ, var.b_inequ, var.w_equ, var.b_equ, 
                     var.states.state_len + var.num_extra_constraints,  // ???
                     batch_size);
 
        // pivoting
        vertPivoting(faces_buffer, var.vertices_buffer, var.VERT_MAX, var.dist_idx, var.As, var.bs, var.xs, var.cnt_x, 
                     var.SOLVE_SIZE_MAX, var.tabij, var.w_inequ, var.b_inequ, var.w_equ, var.b_equ, 
                     var.states.state_len + var.num_extra_constraints,  // ???
                     batch_size);
        
        // correct normals
        corrNormDir(var.vertices_buffer, var.VERT_MAX, var.w_equ, batch_size, flip_insideout);

        // infer new states
        inferAndAppend(states_buffer, batch_size, var);

        // save results
        var.polymesh.append_verts(var.vertices_buffer, batch_size);

    }

    has_am = true;
}


void AnalyticMarching_cuda(const std::vector<torch::Tensor> &weights,
                           const std::vector<torch::Tensor> &biases,
                           const torch::Tensor &states,
                           const torch::Tensor &points,
                           const std::vector<torch::Tensor> &arc_tm,
                           const torch::Tensor &w_extra_constraints,
                           const torch::Tensor &b_extra_constraints,
                           double iso,
                           bool flip_insideout)
{
    has_am = false;
    has_calc = false;

    AT_DISPATCH_FLOATING_TYPES(weights[0].scalar_type(), "main_", ([&] 
        {
            main_<scalar_t>(weights, biases, states, points, arc_tm, w_extra_constraints, b_extra_constraints, iso, flip_insideout, *(var_ptr<scalar_t>));
        }
    ));
    
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

void Init_cuda(const std::string& float_type, const std::vector<int>& nodesnum, const torch::Tensor& arc_table, int num_extra_constraints)
{
    if(float_type == "float32")
    {
        var_ptr<float> = new VAR<float> ();
        var_ptr<float>->mlp.init(arc_table, nodesnum); 
        var_ptr<float>->states.init(std::accumulate(nodesnum.begin()+1, nodesnum.end()-1, 0), var_ptr<float>->BATCH_SIZE_MAX); 
        var_ptr<float>->polymesh.init();
        var_ptr<float>->Init(num_extra_constraints);
    }
    else if (float_type == "float64")
    {
        var_ptr<double> = new VAR<double> ();
        var_ptr<double>->mlp.init(arc_table, nodesnum); 
        var_ptr<double>->states.init(std::accumulate(nodesnum.begin()+1, nodesnum.end()-1, 0), var_ptr<double>->BATCH_SIZE_MAX); 
        var_ptr<double>->polymesh.init();
        var_ptr<double>->Init(num_extra_constraints);
    }

    FloatTypeString = float_type;
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

void Destroy()
{
    if(FloatTypeString == "float32")
    {
        var_ptr<float>->mlp.cudaFreeMemory();
        var_ptr<float>->states.cudaFreeMemory();
        var_ptr<float>->polymesh.cudaFreeMemory();
        var_ptr<float>->Destroy();
        delete var_ptr<float>;
        var_ptr<float> = nullptr;
    }
    else if (FloatTypeString == "float64")
    {
        var_ptr<double>->mlp.cudaFreeMemory();
        var_ptr<double>->states.cudaFreeMemory();
        var_ptr<double>->polymesh.cudaFreeMemory();
        var_ptr<double>->Destroy();
        delete var_ptr<double>;
        var_ptr<double> = nullptr;
    }else
    {
        std::cout << "Environment must be initialized first!" << std::endl;
        return;
    }

    FloatTypeString = "";
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////


void CombineMesh(double scale, std::vector<double> center)
{
    if((FloatTypeString == "float32" || FloatTypeString == "float64") && has_am)
    {
        if(FloatTypeString=="float32")
            var_ptr<float>->polymesh.combine_mesh(scale, center[0], center[1], center[2]);
        else
            var_ptr<double>->polymesh.combine_mesh(scale, center[0], center[1], center[2]);
    }else
    {
        std::cout << "AnalyticMarching must be done first!" << std::endl;
        return;
    }

    has_calc = true;
}

/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////

void ExportMesh(std::string file_path, bool is_polymesh, bool is_float32)
{
    if((FloatTypeString == "float32" || FloatTypeString == "float64") && has_calc)
    {
        if(is_polymesh)
        {
            if(is_float32)
            {
                if(FloatTypeString=="float32")
                    var_ptr<float>->polymesh.export_mesh<float, true>(file_path);
                else
                    var_ptr<double>->polymesh.export_mesh<float, true>(file_path);
            }else
            {
                if(FloatTypeString=="float32")
                    var_ptr<float>->polymesh.export_mesh<double, true>(file_path);
                else
                    var_ptr<double>->polymesh.export_mesh<double, true>(file_path);
            }
        }else
        {
            if(is_float32)
            {
                if(FloatTypeString=="float32")
                    var_ptr<float>->polymesh.export_mesh<float, false>(file_path);
                else
                    var_ptr<double>->polymesh.export_mesh<float, false>(file_path);
            }else
            {
                if(FloatTypeString=="float32")
                    var_ptr<float>->polymesh.export_mesh<double, false>(file_path);
                else
                    var_ptr<double>->polymesh.export_mesh<double, false>(file_path);
            }
        }
    }
    else
    {
        std::cout << "CombineMesh must be done first!" << std::endl;
        return;
    }
}
