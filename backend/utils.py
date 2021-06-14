import os
import torch
import tempfile

def get_boundary(boundary_type='cube', **kwargs):
    """get boundary plane for primitive

    Args:
        boundary_type (str, optional): specify the primitive type. 
            Defaults to 'cube'.
            Options are 
                - 'cube': must specify `min_vert` and `max_vert` in kwargs
        kwargs: 
            min_vert (tuple, optional): min vertex of cube. for example, min_vert = (-1, -1, -1)
            max_vert (tuple, optional): max vertex of cube. for example, max_vert = (1, 1, 1)
    Returns:
        w (torch.Tensor): directions of planes
        b (torch.Tensor): offsets of planes
    """
    if boundary_type == 'cube':
        max_vert = kwargs['max_vert']
        min_vert = kwargs['min_vert']
        w = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=torch.float32)
        b = torch.tensor([-max_vert[0], -max_vert[1], -max_vert[2], min_vert[0], min_vert[1], min_vert[2]],
                         dtype=torch.float32)
    else:
        raise Exception(f'Error: No such {boundary_type}')

    return w, b

def simplify(ply_path, save_ply_path=None, target_perc=0.01, meshlabserver_path="meshlabserver"):
    """ simplify mesh (load ply_path, simplify, and save to save_ply_path)
    """
    script = \
    """
<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Simplification: Quadric Edge Collapse Decimation">
  <Param name="TargetFaceNum" description="Target number of faces" value="100000" type="RichInt" tooltip="The desired final number of faces."/>
  <Param name="TargetPerc" description="Percentage reduction (0..1)" value="{TargetPerc}" type="RichFloat" tooltip="If non zero, this parameter specifies the desired final size of the mesh as a percentage of the initial size."/>
  <Param name="QualityThr" description="Quality threshold" value="0.5" type="RichFloat" tooltip="Quality threshold for penalizing bad shaped faces.&lt;br>The value is in the range [0..1]&#xa; 0 accept any kind of face (no penalties),&#xa; 0.5  penalize faces with quality &lt; 0.5, proportionally to their shape&#xa;"/>
  <Param name="PreserveBoundary" description="Preserve Boundary of the mesh" value="false" type="RichBool" tooltip="The simplification process tries to do not affect mesh boundaries during simplification"/>
  <Param name="BoundaryWeight" description="Boundary Preserving Weight" value="1" type="RichFloat" tooltip="The importance of the boundary during simplification. Default (1.0) means that the boundary has the same importance of the rest. Values greater than 1.0 raise boundary importance and has the effect of removing less vertices on the border. Admitted range of values (0,+inf). "/>
  <Param name="PreserveNormal" description="Preserve Normal" value="true" type="RichBool" tooltip="Try to avoid face flipping effects and try to preserve the original orientation of the surface"/>
  <Param name="PreserveTopology" description="Preserve Topology" value="false" type="RichBool" tooltip="Avoid all the collapses that should cause a topology change in the mesh (like closing holes, squeezing handles, etc). If checked the genus of the mesh should stay unchanged."/>
  <Param name="OptimalPlacement" description="Optimal position of simplified vertices" value="true" type="RichBool" tooltip="Each collapsed vertex is placed in the position minimizing the quadric error.&#xa; It can fail (creating bad spikes) in case of very flat areas. &#xa;If disabled edges are collapsed onto one of the two original vertices and the final mesh is composed by a subset of the original vertices. "/>
  <Param name="PlanarQuadric" description="Planar Simplification" value="true" type="RichBool" tooltip="Add additional simplification constraints that improves the quality of the simplification of the planar portion of the mesh, as a side effect, more triangles will be preserved in flat areas (allowing better shaped triangles)."/>
  <Param name="PlanarWeight" description="Planar Simp. Weight" value="0.001" type="RichFloat" tooltip="How much we should try to preserve the triangles in the planar regions. If you lower this value planar areas will be simplified more."/>
  <Param name="QualityWeight" description="Weighted Simplification" value="false" type="RichBool" tooltip="Use the Per-Vertex quality as a weighting factor for the simplification. The weight is used as a error amplification value, so a vertex with a high quality value will not be simplified and a portion of the mesh with low quality values will be aggressively simplified."/>
  <Param name="AutoClean" description="Post-simplification cleaning" value="true" type="RichBool" tooltip="After the simplification an additional set of steps is performed to clean the mesh (unreferenced vertices, bad faces, etc)"/>
  <Param name="Selected" description="Simplify only selected faces" value="false" type="RichBool" tooltip="The simplification is applied only to the selected set of faces.&#xa; Take care of the target number of faces!"/>
 </filter>
 <filter name="Select non Manifold Edges "/>
 <filter name="Delete Selected Faces"/>
 <filter name="Close Holes">
  <Param name="MaxHoleSize" description="Max size to be closed " value="100" type="RichInt" tooltip="The size is expressed as number of edges composing the hole boundary"/>
  <Param name="Selected" description="Close holes with selected faces" value="false" type="RichBool" tooltip="Only the holes with at least one of the boundary faces selected are closed"/>
  <Param name="NewFaceSelected" description="Select the newly created faces" value="true" type="RichBool" tooltip="After closing a hole the faces that have been created are left selected. Any previous selection is lost. Useful for example for smoothing the newly created holes."/>
  <Param name="SelfIntersection" description="Prevent creation of selfIntersecting faces" value="true" type="RichBool" tooltip="When closing an holes it tries to prevent the creation of faces that intersect faces adjacent to the boundary of the hole. It is an heuristic, non intersetcting hole filling can be NP-complete."/>
 </filter>
</FilterScript>
    """
    script = script.format(TargetPerc=target_perc)
    tmp_dir = tempfile.mkdtemp()

    script_path = os.path.join(tmp_dir, "script.mlx")
    with open(script_path, "w") as f:
        f.write(script)

    save_ply_path = ply_path if save_ply_path is None else save_ply_path
    os.system(f"{meshlabserver_path} -i {ply_path} -o {save_ply_path} -s {script_path}")

    os.remove(script_path)
    os.rmdir(tmp_dir)

def estimate_am_time(model):
    """ estimate the time required for analytic marching

    Args:
        model (torch.nn.Module): our specified nn module (i.e. loaded by load_model function)
    Returns:
        time (float): estimated am_time (sec.)
    """
    nodes = model.nodes
    depth = len(nodes) - 2
    width = sum(nodes[1:-1]) / depth
    a, b, c, d, e, f = 1.94452188,  0.13816182, -0.14536181,  0.59338494, -1.20459825, 1.17841059
    fn = lambda l, n: (a * n) ** (b * l + c) * (n ** d) * (l ** e) * f
    time = fn(depth, width)
    return time
