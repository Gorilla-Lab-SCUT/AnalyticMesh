from .polymesh import PolyMesh


def poly2tri(src_file, dst_file):
    ''' convert the polygonal mesh to triangular one '''
    mesh = PolyMesh(src_file)
    mesh.poly2tri()
    mesh.save(dst_file)
