import os
import time
import torch
import atexit
import numpy as np
import random
import tempfile
from .utils import get_boundary
from .libpolytools import PolyMesh
from functools import reduce

import importlib
cuamlib = importlib.import_module('build.cuam')

SPHERE_TRACING_DICT = {
    'method': 'sphere_tracing',
    'args': {
        'init_num': 1024,
        'step_size_max': 1.0,
        'step_size_mul': 0.5,
        'step_size_min': 1e-3,
        'init_ball_radius': 1.0,
        'avg_eps': 1e-3,
        'hist_len': 100,
        'corr_min': 0.1,
        'time_out': 60,
    }
}

GRADIENT_DESCENT_DICT = {
    'method': 'gradient_descent',
    'args': {
        'init_num': 1024,
        'lr_max': 1e-2,
        'lr_mul': 0.3,
        'lr_min': 1e-5,
        'corr_min': 0.1,
        'hist_len': 100,
        'loss_type': 'l1',
        'optimizer_type': 'adam',
        'accept_bad': True,
        'init_ball_radius': 1.0,
        'batch_mul': 2,
        'avg_eps': 1e-3,
        'time_out': 60,
    }
}

DICHOTOMY_DICT = {
    'method': 'dichotomy',
    'args': {
        'init_num': 1024,
        'try_pts_num': 4096,
        'init_ball_radius': 1.0,
        'iter_max': 100,
        'avg_eps': 1e-3,
        'time_out': 60,
        'provided_surfpts': None,
        'provided_surfstd': None,
    }
}

VOXEL_DICT = {
    'voxel_size': 0.1,
}

########################################################################################################################


def constraints_filter(x, w_extra_constraints, b_extra_constraints, requires_index=False):
    if (w_extra_constraints is not None) and (b_extra_constraints is not None):
        if w_extra_constraints.shape[0] > 0 and x.shape[0] > 0:
            w_extra_constraints = w_extra_constraints.to(dtype=x.dtype).to(device=x.device)
            b_extra_constraints = b_extra_constraints.to(dtype=x.dtype).to(device=x.device)
            temp = x @ w_extra_constraints.t() + b_extra_constraints.reshape([1, -1])
            index = torch.prod((temp < 0).to(dtype=torch.int32), dim=1).to(dtype=torch.bool)
            if requires_index:
                return x[index], index
            return x[index]
    return x


def init_within_ball(init_num, init_ball_radius, w_extra_constraints=None, b_extra_constraints=None):
    points = torch.zeros([0, 3], dtype=torch.float32)
    while points.size(0) < init_num:
        p = (torch.rand([init_num - points.size(0), 3], dtype=torch.float32) * 2 - 1) * init_ball_radius
        p = p[(p**2).sum(dim=1) < init_ball_radius**2]
        if (w_extra_constraints is not None) and (b_extra_constraints is not None):
            p = constraints_filter(p, w_extra_constraints, b_extra_constraints)
        points = torch.cat([points, p], dim=0)
    return points


class Corr:
    def __init__(self, hist_len):
        self.hist_len = hist_len
        self.arange = np.arange(hist_len)
        self.arr = []

    def push(self, value):
        if len(self.arr) == 0:
            self.arr = [value] * self.hist_len
            return 1.0
        else:
            self.arr = [value] + self.arr[:-1]
            return np.corrcoef(np.array(self.arr), self.arange)[0, 1]


###############################################


def sphere_tracing(model, iso, init_num, step_size_max, step_size_mul, step_size_min, w_extra_constraints,
                   b_extra_constraints, init_ball_radius, avg_eps, hist_len, corr_min, time_out):
    ''' `sphere_tracing` to get initialized points

    Using sphere-tracing-based method to get initialized points. 
    It is suitable for Signed Distance Field which has nearly constant gradient value.

    Notes:
        final num of points may be fewer than `init_num`
        
    '''
    step_size = step_size_max
    okey = False
    start_time = time.time()
    corr_calculator = Corr(hist_len)
    while True:
        # init points
        points = init_within_ball(init_num, init_ball_radius, w_extra_constraints, b_extra_constraints).cuda()

        avg_err_old = 1e10

        # update
        while True:
            points = points.requires_grad_(True)

            pred = model(points) - iso

            avg_err = torch.abs(pred).mean()
            corr = corr_calculator.push(avg_err.item())
            print(f'(cuam) (sphere_init) avg_err = {avg_err}, corr = {corr}, step_size = {step_size}')

            if time.time() - start_time > time_out:
                raise Exception(f'Error: sphere_tracing cannot find solution within time={time_out}!')

            if avg_err > avg_err_old * 2 or (not torch.isfinite(avg_err)) or (avg_err > 100) or (corr < corr_min) or (
                    not np.isfinite(corr)):
                print(f'(cuam) (sphere_init) (step_size=={step_size:.2e}-->{step_size * step_size_mul:.2e}) reinit!')
                step_size = step_size * step_size_mul
                if step_size < step_size_min:
                    raise Exception('Error: sphere_tracing cannot find solution!')
                break
            else:
                avg_err_old = avg_err

            if avg_err < avg_eps:
                okey = True
                break

            grad = torch.autograd.grad(pred,
                                       points,
                                       grad_outputs=torch.ones(pred.size()).cuda(),
                                       retain_graph=True,
                                       create_graph=True,
                                       only_inputs=True)[0]

            with torch.no_grad():
                points = points - step_size * pred * grad

        if okey:
            break

    # return
    points = points.requires_grad_(False)
    points = points[(points**2).sum(dim=1) < init_ball_radius**2]
    points = constraints_filter(points, w_extra_constraints, b_extra_constraints)
    return points  # cuda & float32


def gradient_descent(model, iso, init_num, lr_max, lr_mul, lr_min, corr_min, hist_len, loss_type, optimizer_type,
                     accept_bad, w_extra_constraints, b_extra_constraints, init_ball_radius, batch_mul, avg_eps,
                     time_out):
    ''' `gradient_descent` to get initialized points

    Using gradient descent to get initialized points. It may be slow.

    Notes:
        final num of points should be exactly equal to `init_num`
        
    '''
    def smooth_l1(x):
        return torch.where(torch.abs(x) < 1, 0.5 * (x**2), torch.abs(x) - 0.5)

    if loss_type == 'smooth_l1':
        A = avg_eps * 10
        gd_loss = lambda x: (smooth_l1(x / A) * A).mean()
    elif loss_type == 'l1':
        gd_loss = lambda x: torch.abs(x).mean()
    elif loss_type == 'l2':
        gd_loss = lambda x: (x**2).mean()  # ??????????????????
    else:
        raise Exception(f"Error: No such {loss_type}")

    time_start = time.time()
    all_p = torch.zeros([0, 3]).cuda()
    while all_p.size(0) < init_num:
        p = init_within_ball(int((init_num - all_p.size(0)) * batch_mul), init_ball_radius, w_extra_constraints,
                             b_extra_constraints).cuda().requires_grad_()
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam([p], lr=lr_max)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD([p], lr=lr_max, momentum=0.9)
        else:
            raise Exception(f"Error: No such {optimizer_type}")

        pred_err_mean = 1e9
        is_first_time = True
        corr_calculator = Corr(hist_len)
        while pred_err_mean > avg_eps:
            if not is_first_time:
                loss = gd_loss(pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pred = model(p) - iso
            pred_err_mean = torch.abs(pred).mean().item()
            corr = corr_calculator.push(pred_err_mean)
            print(
                f'(cuam) (gradient_descent) pred_err_mean = {pred_err_mean:.2e} | corr = {corr:.2e} | lr = {optimizer.param_groups[0]["lr"]:.2e}'
            )
            is_first_time = False

            if time.time() - time_start > time_out:
                raise Exception(f'Error: gradient_descent cannot find solution within {time_out} seconds!')

            if corr < corr_min:
                corr_calculator = Corr(hist_len)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * lr_mul
                if optimizer.param_groups[0]["lr"] < lr_min:
                    break

        if optimizer.param_groups[0]["lr"] >= lr_min or accept_bad:
            p = constraints_filter(p, w_extra_constraints, b_extra_constraints)
            if p.size(0) > init_num - all_p.size(0):
                p = p[:(init_num - all_p.size(0))]
            all_p = torch.cat([all_p, p], dim=0)

    return all_p


def dichotomy(
        model,
        iso,
        init_num,
        w_extra_constraints,
        b_extra_constraints,
        try_pts_num,  # distance between two adjacent points
        init_ball_radius,  # within this radius
        iter_max,
        avg_eps,
        time_out,
        provided_surfpts=None,  # CPU Tensor. if provided, samples are from nearby surface region.
        provided_surfstd=None,  # float. if provided, samples are from nearby surface region.
):
    ''' `dichotomy` to get initialized points

    Using dichotomy to get initialized points. 
    It is very suitable for Occupancy Field which has bad gradient property.
    Generally, it is suitable for all implicit field.

    It first initializes some pairs of points within a ball,
    then apply dichotomy iteratively to solve the zero crossing points.

    Notes:
        final num of points should be exactly equal to `init_num`
    '''
    def get_pairs(fn, num_pairs, try_pts_num, within_ball_radius):
        ''' pairs are generated by random sampling and finding opposite sign points '''
        time_start = time.time()
        pt_neg = torch.zeros([0, 3]).cuda()
        pt_pos = torch.zeros([0, 3]).cuda()
        while pt_neg.size(0) < num_pairs:
            if time.time() - time_start > time_out:
                raise Exception(f'Error: dichotomy cannot find solution within {time_out} seconds!')

            if (provided_surfpts is not None) and (provided_surfstd is not None):
                points = (provided_surfpts + provided_surfstd * torch.randn_like(provided_surfpts)).cuda()
            else:
                points = init_within_ball(try_pts_num, within_ball_radius, w_extra_constraints,
                                          b_extra_constraints).cuda()
            values = fn(points).reshape(-1) - iso
            id_pos = torch.where(values > 0)[0]
            id_neg = torch.where(values < 0)[0]
            n_pos = len(id_pos)
            n_neg = len(id_neg)
            if n_pos != 0 and n_neg != 0:
                n_range = n_pos * n_neg
                n_sample = (num_pairs - pt_neg.size(0)) if (num_pairs - pt_neg.size(0)) <= n_range else n_range
                index = torch.tensor(random.sample(range(n_range), n_sample))
                index_neg = (index / n_pos).long()
                index_pos = index % n_pos
                pt_pos = torch.cat([pt_pos, points[id_pos[index_pos]]], dim=0)
                pt_neg = torch.cat([pt_neg, points[id_neg[index_neg]]], dim=0)
                print(f'(cuam) (dichotomy) n_pos = {n_pos} | n_neg = {n_neg}')

        return pt_neg, pt_pos

    pt_neg, pt_pos = get_pairs(model, init_num, try_pts_num, init_ball_radius)
    pt_mid = (pt_neg + pt_pos) / 2

    iter_i = 0
    while iter_i < iter_max:
        va_mid = model(pt_mid).reshape(-1) - iso
        avg_err = torch.abs(va_mid).mean()
        print(f'(cuam) (dichotomy) avg_err = {avg_err}')
        if avg_err < avg_eps:
            break
        va_mid_index = (va_mid < 0)
        va_mid_not_index = ~va_mid_index
        pt_neg[va_mid_index] = pt_mid[va_mid_index]
        pt_pos[va_mid_not_index] = pt_mid[va_mid_not_index]
        pt_mid = (pt_neg + pt_pos) / 2
        iter_i += 1

    return pt_mid  # cuda & float32


########################################################################################################################

ENVIRONMENT_STR = ''


# return includes key: init_time, am_time, export_time (sec.)
def AnalyticMarching(
    model,
    save_ply_path='mesh.ply',
    iso=0.0,
    scale=1.0,
    center=[0.0, 0.0, 0.0],
    w_extra_constraints_=torch.zeros([0, 3]),
    b_extra_constraints_=torch.zeros([0]),
    init_configs=DICHOTOMY_DICT,  # default choice
    save_polymesh=True,
    save_float32_verts=True,
    flip_insideout=False,
    dtype=torch.float64,
    voxel_configs=None,
):
    ''' Analytic Marching (CUDA)
        it save result directly to disk (specified by `save_ply_path`)

    Args:
        model (nn.Module): the mlp model (please refer to `cuam.MLP`)
        save_ply_path (str): path to save .ply mesh file
        iso (float): iso level
        scale (float): scale (it performs: out = in * scale + center)
        center (list or tuple): center (it performs: out = in * scale + center)
        w_extra_constraints_ (torch.Tensor): extra boundary plane constraints (weight)
        b_extra_constraints_ (torch.Tensor): extra boundary plane constraints (bias)
        init_configs (dict): configuration of initialization 
        save_polymesh (bool): save polygon mesh to file, otherwise we save in triangle mesh format
        save_float32_verts (bool): save vertices in float32 format, otherwise we save in float64 format
        flip_insideout (bool): flip insideout, you should set it to True if interior is positive
        dtype (torch.dtype): the used data type when solving analytically, torch.float32 or torch.float64
        voxel_configs (None or dict): configuration of local grid AM
    Returns:
        return_dict (dict): the dict containing consuming time
            it has keys:
                "init_point_time": time to init points
                "init_cuda_time": time to init cuda environment
                "am_time": time to perform analytic marching
                "export_time": time to export file to disk
    '''

    global ENVIRONMENT_STR

    # check
    assert w_extra_constraints_.shape[0] == b_extra_constraints_.shape[0]
    num_extra_constraints = w_extra_constraints_.shape[0] + (0 if voxel_configs is None else 6)
    CONSTRAINTS_JITTER = (1e-8)

    # move to cuda
    model = model.cuda()
    model = model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # init points
    init_configs['args']['w_extra_constraints'] = w_extra_constraints_
    init_configs['args']['b_extra_constraints'] = b_extra_constraints_

    return_dict = {}

    t_start = time.time()
    if init_configs['method'] == 'sphere_tracing':
        points = sphere_tracing(model, iso, **init_configs['args'])
    elif init_configs['method'] == 'dichotomy':
        points = dichotomy(model, iso, **init_configs['args'])
    elif init_configs['method'] == 'gradient_descent':
        points = gradient_descent(model, iso, **init_configs['args'])
    else:
        raise Exception(f"Error: No such {init_configs['method']}")
    return_dict['init_point_time'] = time.time() - t_start
    print(f"(cuam) init_point_time = {return_dict['init_point_time']}")

    # calc states
    with torch.no_grad():
        model(points, requires_outputs_list=True)
        states = torch.cat([(s > 0) for s in model.outputs_list], dim=1)
        model.outputs_list = []

    # info
    info = model.get_info()

    float_type = 'float64' if dtype == torch.float64 else 'float32'
    nodesnum = model.nodes
    arc_table = info['arc_table'].to(dtype=torch.int32).cpu()

    weights = [w.to(dtype=dtype).cuda() for w in info['weights']]
    biases = [b.to(dtype=dtype).cuda() for b in info['biases']]
    states = states.to(dtype=torch.bool).cpu()
    points = points.to(dtype=dtype).cpu()
    arc_tm = [t.to(dtype=dtype).cuda() for t in info['arc_tm']]
    w_extra_constraints = w_extra_constraints_.to(dtype=dtype).cuda()
    b_extra_constraints = b_extra_constraints_.to(dtype=dtype).cuda()
    w_extra_constraints += CONSTRAINTS_JITTER  # add small jitter to avoid precision problems

    environment_str = f"float_type = {float_type}\n"
    environment_str += f"nodesnum = {nodesnum}\n"
    environment_str += f"arc_table = {arc_table}"

    ##############################################
    ##############################################

    # init environment
    if environment_str != ENVIRONMENT_STR:
        t_start = time.time()
        if ENVIRONMENT_STR != '':
            cuamlib.Destroy()
        cuamlib.Init(float_type=float_type,
                     nodesnum=nodesnum,
                     arc_table=arc_table,
                     num_extra_constraints=num_extra_constraints)
        return_dict['init_cuda_time'] = time.time() - t_start
        ENVIRONMENT_STR = environment_str
    else:
        return_dict['init_cuda_time'] = 0.0
    print(f"(cuam) init_cuda_time = {return_dict['init_cuda_time']}")

    if voxel_configs is None:
        # execution
        states = states.cuda()
        points = points.cuda()
        t_start = time.time()
        cuamlib.AnalyticMarching(weights=weights,
                                 biases=biases,
                                 states=states,
                                 points=points,
                                 arc_tm=arc_tm,
                                 w_extra_constraints=w_extra_constraints,
                                 b_extra_constraints=b_extra_constraints,
                                 iso=iso,
                                 flip_insideout=flip_insideout)
        return_dict['am_time'] = time.time() - t_start
        print(f"(cuam) am_time = {return_dict['am_time']}")

        # export mesh file
        t_start = time.time()
        cuamlib.CombineMesh(scale=scale, center=center)  # new = old * scale + center
        cuamlib.ExportMesh(file_path=save_ply_path, is_polymesh=save_polymesh, is_float32=save_float32_verts)
        return_dict['export_time'] = time.time() - t_start
        print(f"(cuam) export_time  = {return_dict['export_time']}")

    else:
        voxel_size = voxel_configs['voxel_size']
        index_3d = torch.floor(points / voxel_size).to(dtype=torch.long)
        index_max, _ = index_3d.max(dim=0)
        index_min, _ = index_3d.min(dim=0)
        index_3d = index_3d - index_min
        index_dim = (index_max - index_min + 1)
        index_1d = index_3d[:, 0] * index_dim[1] * index_dim[2] + \
                   index_3d[:, 1] * index_dim[2] + \
                   index_3d[:, 2]
        unique_index_1d = torch.unique(index_1d)

        am_time_list = list()
        export_time_list = list()
        meshes_list = list()
        tmp_dir = tempfile.mkdtemp()
        print(f"(cuam) we are using tmp_dir = {tmp_dir}")
        torch.cuda.empty_cache()
        for grid_id, the_index_1d in enumerate(unique_index_1d):
            select_index = torch.where(index_1d == the_index_1d)
            sub_points = points[select_index].contiguous().cuda()
            sub_states = states[select_index].contiguous().cuda()
            decode_index_3d_0 = the_index_1d / (index_dim[1] * index_dim[2])
            decode_index_3d_1 = (the_index_1d % (index_dim[1] * index_dim[2])) / index_dim[2]
            decode_index_3d_2 = the_index_1d % index_dim[2]
            decode_index_3d_0 = decode_index_3d_0 + index_min[0]
            decode_index_3d_1 = decode_index_3d_1 + index_min[1]
            decode_index_3d_2 = decode_index_3d_2 + index_min[2]
            w_, b_ = get_boundary('cube',
                                  min_vert=(decode_index_3d_0 * voxel_size, decode_index_3d_1 * voxel_size,
                                            decode_index_3d_2 * voxel_size),
                                  max_vert=((decode_index_3d_0 + 1) * voxel_size, (decode_index_3d_1 + 1) * voxel_size,
                                            (decode_index_3d_2 + 1) * voxel_size))
            w_ = w_.to(dtype=dtype).cuda()
            b_ = b_.to(dtype=dtype).cuda()
            w_ += CONSTRAINTS_JITTER  # add small jitter to avoid precision problems

            t_start = time.time()
            cuamlib.AnalyticMarching(weights=[w.clone() for w in weights],
                                     biases=[b.clone() for b in biases],
                                     states=sub_states,
                                     points=sub_points,
                                     arc_tm=[a.clone() for a in arc_tm],
                                     w_extra_constraints=torch.cat([w_, w_extra_constraints], dim=0).contiguous(),
                                     b_extra_constraints=torch.cat([b_, b_extra_constraints], dim=0).contiguous(),
                                     iso=iso,
                                     flip_insideout=flip_insideout)
            am_time_list.append(time.time() - t_start)
            print(f"(cuam) [{grid_id}/{len(unique_index_1d)}] am_time = {am_time_list[-1]}")

            # export mesh file
            t_start = time.time()
            cuamlib.CombineMesh(scale=scale, center=center)  # new = old * scale + center
            cuamlib.ExportMesh(file_path=os.path.join(tmp_dir, f"{grid_id}.ply"),
                               is_polymesh=save_polymesh,
                               is_float32=save_float32_verts)
            export_time_list.append(time.time() - t_start)
            print(f"(cuam) [{grid_id}/{len(unique_index_1d)}] export_time  = {export_time_list[-1]}")
            meshes_list.append(PolyMesh(os.path.join(tmp_dir, f"{grid_id}.ply")))

        return_dict['am_time'] = sum(am_time_list)
        return_dict['export_time'] = sum(export_time_list)
        print(f"(cuam) [total] am_time  = {return_dict['am_time']}")
        print(f"(cuam) [total] export_time  = {return_dict['export_time']}")

        combined_mesh_vertices_list = [m.vertices() for m in meshes_list]
        combined_mesh_faces_list = [m.faces() for m in meshes_list]
        combined_mesh_faces = list()
        combined_mesh_vertices = reduce(lambda x, y: x + y, combined_mesh_vertices_list)
        vertices_sum = 0
        for v, f in zip(combined_mesh_vertices_list, combined_mesh_faces_list):
            f_ = list(map(lambda x: list(map(lambda y: y + vertices_sum, x)), f))
            vertices_sum += len(v)
            combined_mesh_faces.extend(f_)
        combined_mesh = PolyMesh(vertices=combined_mesh_vertices, faces=combined_mesh_faces, colors=[])
        combined_mesh.save(save_ply_path)

        for grid_id in range(len(unique_index_1d)):
            if os.path.exists(os.path.join(tmp_dir, f"{grid_id}.ply")):
                os.remove(os.path.join(tmp_dir, f"{grid_id}.ply"))
        os.rmdir(tmp_dir)
        print(f"(cuam) removed tmp_dir = {tmp_dir}")

    # return
    return return_dict


@atexit.register
def when_exit():
    if ENVIRONMENT_STR != '':
        cuamlib.Destroy()
