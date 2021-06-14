/*
    A simple polygon mesh processing library
*/

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

////////////////////////////////////////////////////////////

std::vector<std::string> split(std::string str, std::string pattern)
{
    std::vector<std::string> res;
    if (str == "")
        return res;
    std::string strs = str + pattern;
    auto pos = strs.find(pattern);

    while (pos != strs.npos)
    {
        res.push_back(strs.substr(0, pos));
        strs = strs.substr(pos + 1, strs.size());
        pos = strs.find(pattern);
    }

    return res;
}

std::string get_extension(std::string file_path)
{
    return file_path.substr(file_path.find_last_of(".") + 1);
}

template <typename T>
class Point
{
public:
    T x, y, z;
    Point() : x(0), y(0), z(0) {}
    Point(T *ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}
    Point(T x_, T y_, T z_) : x(x_), y(y_), z(z_) {}
    bool is_degenerated()
    {
        return (x == y || y == z || z == x);
    }
};

////////////////////////////////////////////////////////////

class PolyMesh
{
private:
    std::vector<Point<float>> _vertices;
    std::vector<std::vector<int>> _faces;
    std::vector<Point<u_char>> _colors; // faces colors

    void load_ply(std::string load_path); // binary, load vertices is float or double
    void load_off(std::string load_path); // ascii
    void save_ply(std::string save_path); // save in float vertices
    void save_off(std::string save_path); // save in ascii

public:
    // basic
    PolyMesh() {}
    PolyMesh(std::string load_path) { load(load_path); }
    PolyMesh(const std::vector<std::vector<float>> &vertices, const std::vector<std::vector<int>> &faces, const std::vector<std::vector<int>> &colors);
    void clear();
    void load(std::string load_path);
    void save(std::string save_path);
    // property
    std::vector<std::vector<float>> vertices();
    std::vector<std::vector<int>> faces();
    std::vector<std::vector<int>> colors();
    void set_vertices(const std::vector<std::vector<float>> &vertices);
    void set_faces(const std::vector<std::vector<int>> &faces);
    void set_colors(const std::vector<std::vector<int>> &colors);
    // function
    void poly2tri();
    bool is_polymesh();
    int num_polyfaces(); // for trimesh, we have: polyfaces_num == trifaces_num
    int num_trifaces();  // for polymesh, we have: trifaces_num >= polyfaces_num
};

////////////////////////////////////////////////////////////

PolyMesh::PolyMesh(const std::vector<std::vector<float>> &vertices, const std::vector<std::vector<int>> &faces, const std::vector<std::vector<int>> &colors)
{
    set_vertices(vertices);
    set_faces(faces);
    set_colors(colors);
}

void PolyMesh::clear()
{
    if (_vertices.size() > 0)
    {
        _vertices.clear();
    }
    if (_faces.size() > 0)
    {
        _faces.clear();
    }
    if (_colors.size() > 0)
    {
        _colors.clear();
    }
}

////////////////////////////////////////////////////////////
//////////////////// Loading && Saving /////////////////////
////////////////////////////////////////////////////////////
void PolyMesh::load(std::string load_path)
{
    auto extension = get_extension(load_path);
    std::for_each(extension.begin(), extension.end(), [](char &c) { c = ::tolower(c); });
    if (extension == "ply")
    {
        load_ply(load_path);
    }
    else if (extension == "off")
    {
        load_off(load_path);
    }
    else
    {
        throw std::runtime_error("Unsupported file format.");
    }
}

void PolyMesh::load_ply(std::string load_path)
{
    clear();

    int vertices_num;
    int faces_num;

    std::ifstream ply_file(load_path, std::ios::in | std::ios::binary);
    std::string line;
    std::string vertices_float_type;

    getline(ply_file, line);
    std::for_each(line.begin(), line.end(), [](char &c) { c = ::tolower(c); });
    if (line != "ply")
    {
        throw std::runtime_error("It should be a ply file!");
    }

    std::string read_style = "";
    int has_colors = 0;
    bool continue_flag = true;
    while (continue_flag)
    {
        getline(ply_file, line);
        std::for_each(line.begin(), line.end(), [](char &c) { c = ::tolower(c); });
        if (line.find("comment") != std::string::npos)
        {
            continue;
        }
        else if (line.find("binary_little_endian") != std::string::npos)
        {
            read_style = "binary_little_endian";
            continue;
        }
        else if (line.find("ascii") != std::string::npos || line.find("binary_big_endian") != std::string::npos)
        {
            throw std::runtime_error("We only support binary_little_endian ply file!");
        }
        else if (line.find("vertex") != std::string::npos)
        {
            auto sp_cont = split(line, " ");
            vertices_num = stoi(sp_cont[2]);
            getline(ply_file, line); // property float x
            getline(ply_file, line); // property float y
            getline(ply_file, line); // property float z
            if (line.find("float") != std::string::npos)
            {
                vertices_float_type = "float";
            }
            else if (line.find("double") != std::string::npos)
            {
                vertices_float_type = "double";
            }
            else
            {
                throw std::runtime_error("Unknown vertices type.");
            }
        }
        else if (line.find("face") != std::string::npos)
        {
            auto sp_cont = split(line, " ");
            faces_num = stoi(sp_cont[2]);
            getline(ply_file, line); // property list uchar int vertex_index
            has_colors += 1;
        }
        else if (line.find("end_header") != std::string::npos)
        {
            continue_flag = false;
        }
        else if (has_colors != 0 && (line.find("red") != std::string::npos ||
                                     line.find("green") != std::string::npos ||
                                     line.find("blue") != std::string::npos))
        {
            has_colors += 1;
        }
        else
        {
            std::cout << "Unknown line: " << line << std::endl;
            continue;
        }
    }

    if (read_style != "binary_little_endian")
    {
        throw std::runtime_error("Unknown storage format, but we only support `binary_little_endian`.");
    }

    if (!(has_colors == 1 || has_colors == 4))
    {
        throw std::runtime_error("Incomplete colors.");
    }

    //////////////////
    if (vertices_float_type == "float")
    {
        float array3[3];
        for (int i = 0; i < vertices_num; ++i)
        {
            ply_file.read((char *)array3, sizeof(float) * 3);
            _vertices.push_back(Point<float>(array3));
        }
    }
    else
    {
        double array3[3];
        for (int i = 0; i < vertices_num; ++i)
        {
            ply_file.read((char *)array3, sizeof(double) * 3);
            _vertices.push_back(Point<float>(float(array3[0]), float(array3[1]), float(array3[2])));
        }
    }

    u_char f_v_num;
    int vi;
    u_char red, green, blue;
    for (int i = 0; i < faces_num; ++i)
    {
        ply_file.read((char *)&f_v_num, sizeof(u_char));
        _faces.push_back(std::vector<int>());
        for (int j = 0; j < f_v_num; ++j)
        {
            ply_file.read((char *)&vi, sizeof(int));
            _faces.back().push_back(vi);
        }
        if (has_colors == 4)
        {
            ply_file.read((char *)&red, sizeof(u_char));
            ply_file.read((char *)&green, sizeof(u_char));
            ply_file.read((char *)&blue, sizeof(u_char));
            _colors.push_back(Point<u_char>(red, green, blue));
        }
    }

    ply_file.close();
}

void PolyMesh::load_off(std::string load_path)
{
    clear();

    int vertices_num;
    int faces_num;

    std::ifstream off_file(load_path, std::ios::in); // ascii
    std::string line;

    getline(off_file, line);
    std::for_each(line.begin(), line.end(), [](char &c) { c = ::tolower(c); });
    if (line != "off")
    {
        throw std::runtime_error("It should be an ascii off file!");
    }

    getline(off_file, line);
    auto sp_cont = split(line, " ");
    vertices_num = stoi(sp_cont[0]);
    faces_num = stoi(sp_cont[1]);

    //////////////////

    float x, y, z;
    for (int i = 0; i < vertices_num; ++i)
    {
        getline(off_file, line);
        auto sp_cont = split(line, " ");
        x = stof(sp_cont[0]);
        y = stof(sp_cont[1]);
        z = stof(sp_cont[2]);
        _vertices.push_back(Point<float>(x, y, z));
    }

    int f_v_num, vi;
    u_char red, green, blue;
    for (int i = 0; i < faces_num; ++i)
    {
        getline(off_file, line);
        auto sp_cont = split(line, " ");
        f_v_num = stoi(sp_cont[0]);

        _faces.push_back(std::vector<int>());
        for (int j = 0; j < f_v_num; ++j)
        {
            vi = stoi(sp_cont[1 + j]);
            _faces.back().push_back(vi);
        }
        if (sp_cont.size() > f_v_num + 1)
        {
            red = int(roundf(stof(sp_cont[f_v_num + 1]) * 255));
            green = int(roundf(stof(sp_cont[f_v_num + 2]) * 255));
            blue = int(roundf(stof(sp_cont[f_v_num + 3]) * 255));
            _colors.push_back(Point<u_char>(red, green, blue));
        }
    }

    off_file.close();
}

void PolyMesh::save(std::string save_path)
{
    auto extension = get_extension(save_path);
    std::for_each(extension.begin(), extension.end(), [](char &c) { c = ::tolower(c); });
    if (extension == "ply")
    {
        save_ply(save_path);
    }
    else if (extension == "off")
    {
        save_off(save_path);
    }
    else
    {
        throw std::runtime_error("Unsupported file format.");
    }
}

void PolyMesh::save_ply(std::string save_path)
{
    std::ofstream ply_file(save_path, std::ios::out | std::ios::binary);
    ply_file << "ply\n";
    ply_file << "format binary_little_endian 1.0\n";
    ply_file << "element vertex " << _vertices.size() << std::endl;
    ply_file << "property float x\n";
    ply_file << "property float y\n";
    ply_file << "property float z\n";
    ply_file << "element face " << _faces.size() << std::endl;
    ply_file << "property list uchar int vertex_index\n";
    if (_colors.size())
    {
        ply_file << "property uchar red" << std::endl;
        ply_file << "property uchar green" << std::endl;
        ply_file << "property uchar blue" << std::endl;
    }
    ply_file << "end_header\n";

    for (auto &p : _vertices)
    {
        ply_file.write((char *)&(p.x), sizeof(float));
        ply_file.write((char *)&(p.y), sizeof(float));
        ply_file.write((char *)&(p.z), sizeof(float));
    }

    for (int i = 0; i < _faces.size(); ++i)
    {
        auto &f = _faces[i];
        u_char v_num = (u_char)(f.size());
        ply_file.write((char *)&v_num, sizeof(u_char));

        for (int &i : f)
        {
            ply_file.write((char *)&i, sizeof(int));
        }
        if (_colors.size())
        {
            ply_file.write((char *)&(_colors[i].x), sizeof(u_char));
            ply_file.write((char *)&(_colors[i].y), sizeof(u_char));
            ply_file.write((char *)&(_colors[i].z), sizeof(u_char));
        }
    }
    ply_file.close();
}

void PolyMesh::save_off(std::string save_path)
{
    std::ofstream off_file(save_path, std::ios::out);
    off_file << "OFF" << std::endl;
    off_file << _vertices.size() << " " << _faces.size() << " 0" << std::endl;

    for (auto &p : _vertices)
    {
        off_file << p.x << " " << p.y << " " << p.z << std::endl;
    }

    for (int i = 0; i < _faces.size(); ++i)
    {
        auto &f = _faces[i];

        off_file << f.size();
        for (int &i : f)
        {
            off_file << " " << i;
        }
        if (_colors.size())
        {
            off_file << " " << _colors[i].x << " " << _colors[i].y << " " << _colors[i].z;
        }
        off_file << std::endl;
    }
    off_file.close();
}

////////////////////////////////////////////////////////////

std::vector<std::vector<float>> PolyMesh::vertices()
{
    std::vector<std::vector<float>> vertices;
    for (auto &v : _vertices)
    {
        vertices.push_back(std::vector<float>({v.x, v.y, v.z}));
    }
    return vertices;
}
std::vector<std::vector<int>> PolyMesh::faces()
{
    return _faces;
}

std::vector<std::vector<int>> PolyMesh::colors()
{
    std::vector<std::vector<int>> colors;
    for (auto &c : _colors)
    {
        colors.push_back(std::vector<int>({c.x, c.y, c.z}));
    }
    return colors;
}

void PolyMesh::set_vertices(const std::vector<std::vector<float>> &vertices)
{
    for (auto &v : vertices)
    {
        if (v.size() != 3)
        {
            throw std::runtime_error("Vertices should be 3-dimensional.");
        }
    }

    _vertices.clear();
    for (auto &v : vertices)
    {
        _vertices.push_back(Point<float>(v[0], v[1], v[2]));
    }
}

void PolyMesh::set_faces(const std::vector<std::vector<int>> &faces)
{
    for (auto &f : faces)
    {
        if (f.size() < 3)
        {
            throw std::runtime_error("Faces should have at least 3 vertices.");
        }
    }

    _faces.clear();
    _faces = faces;
}

void PolyMesh::set_colors(const std::vector<std::vector<int>> &colors)
{
    for (auto &c : colors)
    {
        if (c.size() != 3 ||
            c[0] < 0 || c[0] > 255 ||
            c[1] < 0 || c[1] > 255 ||
            c[2] < 0 || c[2] > 255)
        {
            throw std::runtime_error("RGB colors should be in integer range: [0, 255].");
        }
    }

    _colors.clear();
    for (auto &c : colors)
    {
        _colors.push_back(Point<u_char>(c[0], c[1], c[2]));
    }
}

////////////////////////////////////////////////////////////

void PolyMesh::poly2tri()
{
    int faces_num = _faces.size();
    if (faces_num)
    {
        std::vector<std::vector<int>> faces_out;
        std::vector<Point<u_char>> colors_out;
        int face_vert_num = 0;
        for (int i = 0; i < faces_num; ++i)
        {
            face_vert_num = _faces[i].size();
            for (int j = 0; j < face_vert_num - 2; ++j)
            {
                Point<int> triface(_faces[i][0], _faces[i][j + 1], _faces[i][j + 2]);
                if (!triface.is_degenerated())
                {
                    // delete degenerated faces
                    faces_out.push_back(std::vector<int>({triface.x, triface.y, triface.z}));
                    if (_colors.size())
                    {
                        colors_out.push_back(_colors[i]);
                    }
                }
            }
        }

        _faces.swap(faces_out);
        _colors.swap(colors_out);
    }
}

bool PolyMesh::is_polymesh()
{
    if (_faces.size() == 0)
    {
        return true;
    }

    for (auto &f : _faces)
    {
        if (f.size() > 3)
        {
            return true;
        }
    }
    return false;
}

int PolyMesh::num_polyfaces()
{
    return _faces.size();
}

int PolyMesh::num_trifaces()
{
    if (is_polymesh())
    {
        int cnt = 0;
        for (auto &f : _faces)
        {
            cnt += (f.size() - 2); // NOTE: we do not delete degenerated faces
        }
        return cnt;
    }
    else
    {
        return _faces.size();
    }
}

////////////////////////////////////////////////////////

PYBIND11_MODULE(polylib, m)
{
    m.doc() = "Python library for polygon mesh processing";

    pybind11::class_<PolyMesh>(m, "PolyMesh")
        .def(pybind11::init(), "Init for PolyMesh")
        .def(pybind11::init<std::string>(), "Init for PolyMesh", pybind11::arg("load_path"))
        .def(pybind11::init<const std::vector<std::vector<float>> &, const std::vector<std::vector<int>> &, const std::vector<std::vector<int>> &>(), "Init for PolyMesh", pybind11::arg("vertices"), pybind11::arg("faces"), pybind11::arg("colors"))
        .def("clear", &PolyMesh::clear, "Clear up the memory")
        .def("load", &PolyMesh::load, "Load mesh from file (support: binary ply, ascii off)", pybind11::arg("load_path"))
        .def("save", &PolyMesh::save, "Save mesh to file (support: binary ply, ascii off)", pybind11::arg("save_path"))
        .def("vertices", &PolyMesh::vertices, "Get vertices")
        .def("faces", &PolyMesh::faces, "Get faces")
        .def("colors", &PolyMesh::colors, "Get colors")
        .def("set_vertices", &PolyMesh::set_vertices, "Set vertices", pybind11::arg("vertices"))
        .def("set_faces", &PolyMesh::set_faces, "Set faces", pybind11::arg("faces"))
        .def("set_colors", &PolyMesh::set_colors, "Set colors", pybind11::arg("colors"))
        .def("poly2tri", &PolyMesh::poly2tri, "Convert to triangular mesh")
        .def("is_polymesh", &PolyMesh::is_polymesh, "Whether it is polygonal")
        .def("num_polyfaces", &PolyMesh::num_polyfaces, "Get the num of polygonal faces")
        .def("num_trifaces", &PolyMesh::num_trifaces, "Get the num of triangular faces");
}
