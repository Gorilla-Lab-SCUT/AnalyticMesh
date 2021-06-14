""" read header of ply file
"""


def load_ply_header(ply_path):
    """ load header info from ply file (triangle mesh or polygon mesh)

    Args:
        ply_path (str): path to ply file
    Returns:
        info (dict): a dictionary containing the parsed header
            e.g. 
                info == {'storing_type': 'binary_little_endian',
                         'storing_version': 1.0,
                         'vertex_num': 36692453,
                         'face_num': 73159217,
                        }
    """
    # load header
    content = b""
    stride = 1000
    stride_i = 1
    while not ((b"end_header" in content) or (b"END_HEADER" in content)):
        with open(ply_path, "rb") as f:
            content = f.read(stride * stride_i)
        stride_i += 1

    lines_ = content.split(b"\n")
    lines = list()
    for s in lines_:
        try:
            lines.append(s.decode('utf-8'))
        except:
            pass

    # parsing
    info = dict()
    for s_ in lines:
        s = s_.lower()
        if s.startswith("format"):
            info["storing_type"] = s.split(" ")[1]
            info["storing_version"] = float(s.split(" ")[2])
        elif s.startswith("element"):
            what_elem = s.split(" ")[1]
            elem_num = int(s.split(" ")[2])
            info[f"{what_elem}_num"] = elem_num
        elif s.startswith("property"):
            pass
        elif s.startswith("end_header"):
            break

    return info
