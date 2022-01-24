#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<uint32_t> randomWalks(py::array_t<uint32_t> _indptr, py::array_t<uint32_t> _indices, py::array_t<float> _data, py::array_t<uint32_t> _startNodes, size_t nWalks, size_t walkLen);
py::array_t<uint32_t> randomWalksRestart(py::array_t<uint32_t> _indptr, py::array_t<uint32_t> _indices, py::array_t<float> _data, py::array_t<uint32_t> _startNodes, size_t nWalks, size_t walkLen, float alpha);
py::array_t<uint32_t> n2vRandomWalks(py::array_t<uint32_t> _indptr, py::array_t<uint32_t> _indices, py::array_t<float> _data, py::array_t<uint32_t> _startNodes, size_t nWalks, size_t walkLen, float p, float q);
py::array_t<bool> corruptWalks(py::array_t<uint32_t> _walks, size_t nNodes, float r);
py::array_t<bool> weightedCorruptWalks(py::array_t<uint32_t> _walks, py::array_t<uint32_t> _candidates, size_t nNodes, float r);
