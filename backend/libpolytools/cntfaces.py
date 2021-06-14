''' count the num of faces of a mesh '''

from .polymesh import PolyMesh


def get_faces_num(mesh_file):
    ''' obtain the num of faces (polygonal faces and triangular faces)
        input mesh is polygonal or triangular
        for triangular mesh, tri == poly
        for polygonal mesh, tri >= poly
    '''
    mesh = PolyMesh(mesh_file)
    return {'poly': mesh.num_polyfaces(), 'tri': mesh.num_trifaces()}
