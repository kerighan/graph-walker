#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "randomWalks.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_walker, m)
{
    m.def("random_walks", &randomWalks, "random walks");
    m.def("random_walks_with_restart", &randomWalksRestart, "random walks with restart");
    m.def("node2vec_random_walks", &n2vRandomWalks, "node2vec random walks");
    m.def("corrupt", &corruptWalks, "corrupt walks");
    m.def("weighted_corrupt", &weightedCorruptWalks, "weighted corrupt walks");
}
