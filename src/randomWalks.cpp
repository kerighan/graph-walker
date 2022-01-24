#include <iostream>
#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include "fastRandom.hpp"
#include "threading.hpp"
#include "randomWalks.hpp"

namespace py = pybind11;

py::array_t<uint32_t> randomWalks(py::array_t<uint32_t> _indptr, py::array_t<uint32_t> _indices, py::array_t<float> _data, py::array_t<uint32_t> _startNodes, size_t nWalks, size_t walkLen)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info startNodesBuf = _startNodes.request();
    uint32_t *startNodes = (uint32_t *)startNodesBuf.ptr;

    // general variables
    size_t nNodes = startNodesBuf.shape[0];
    size_t shape = nWalks * nNodes;

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 generator(rd());
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step = startNodes[i % nNodes];
        walks[i * walkLen] = step;

        for (size_t k = 1; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we fill in current node
            if (start == end)
            {
                walks[i * walkLen + k] = step;
                continue;
            }

            // searchsorted
            float cumsum = 0;
            size_t index = 0;
            float draw = draws[k - 1];
            for (size_t z = start; z < end; z++)
            {
                cumsum += data[z];
                if (draw > cumsum)
                {
                    continue;
                }
                else
                {
                    index = z;
                    break;
                }
            }

            // draw next index
            step = indices[index];

            // update walk
            walks[i * walkLen + k] = step;
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

py::array_t<uint32_t> randomWalksRestart(
    py::array_t<uint32_t> _indptr,
    py::array_t<uint32_t> _indices,
    py::array_t<float> _data,
    py::array_t<uint32_t> _startNodes,
    size_t nWalks,
    size_t walkLen,
    float alpha)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info startNodesBuf = _startNodes.request();
    uint32_t *startNodes = (uint32_t *)startNodesBuf.ptr;

    // general variables
    size_t nNodes = startNodesBuf.shape[0];
    size_t shape = nWalks * nNodes;

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    // make random walks
    PARALLEL_FOR_BEGIN(shape)
    {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 generator(rd());
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step = startNodes[i % nNodes];
        size_t startNode = step;
        walks[i * walkLen] = step;

        for (size_t k = 1; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we fill in current node
            if (start == end)
            {
                walks[i * walkLen + k] = step;
                continue;
            }

            if (dist(generator) < alpha)
            {
                step = startNode;
            }
            else
            {
                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1];
                for (size_t z = start; z < end; z++)
                {
                    cumsum += data[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }

                // draw next index
                step = indices[index];
            }

            // update walk
            walks[i * walkLen + k] = step;
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

py::array_t<uint32_t> n2vRandomWalks(py::array_t<uint32_t> _indptr, py::array_t<uint32_t> _indices, py::array_t<float> _data, py::array_t<uint32_t> _startNodes, size_t nWalks, size_t walkLen, float p, float q)
{
    // get data buffers
    py::buffer_info indptrBuf = _indptr.request();
    uint32_t *indptr = (uint32_t *)indptrBuf.ptr;

    py::buffer_info indicesBuf = _indices.request();
    uint32_t *indices = (uint32_t *)indicesBuf.ptr;

    py::buffer_info dataBuf = _data.request();
    float *data = (float *)dataBuf.ptr;

    py::buffer_info startNodesBuf = _startNodes.request();
    uint32_t *startNodes = (uint32_t *)startNodesBuf.ptr;

    // general variables
    size_t nNodes = startNodesBuf.shape[0];
    size_t shape = nWalks * nNodes;

    // walk matrix
    py::array_t<uint32_t> _walks({shape, walkLen});
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    PARALLEL_FOR_BEGIN(shape)
    {
        static thread_local std::random_device rd;
        static thread_local std::mt19937 generator(rd());
        std::uniform_real_distribution<> dist(0., 1.);
        std::vector<float> draws;
        draws.reserve(walkLen - 1);
        for (size_t z = 0; z < walkLen - 1; z++)
        {
            draws[z] = dist(generator);
        }

        size_t step = startNodes[i % nNodes];
        walks[i * walkLen] = step;

        for (size_t k = 1; k < walkLen; k++)
        {
            uint32_t start = indptr[step];
            uint32_t end = indptr[step + 1];

            // if no neighbors, we fill in current node
            if (start == end)
            {
                walks[i * walkLen + k] = step;
                continue;
            }

            if (k >= 2)
            {
                uint32_t prev = walks[i * walkLen + k - 2];
                uint32_t prevStart = indptr[prev];
                uint32_t prevEnd = indptr[prev + 1];

                float weightSum = 0.;
                std::vector<float> weights;
                weights.reserve(end - start);
                for (size_t z = start; z < end; z++)
                {
                    uint32_t neighbor = indices[z];
                    float weight = data[z];
                    if (neighbor == prev)
                    {
                        // case where candidate is the previous node
                        weight /= p;
                    }
                    else
                    {
                        // check if candidate is a neighbor of previous node
                        bool isInPrev = false;
                        for (size_t pi = prevStart; pi < prevEnd; pi++)
                        {
                            if (neighbor != indices[pi])
                                continue;
                            isInPrev = true;
                            break;
                        }
                        if (!isInPrev)
                            weight /= q;
                    }
                    weights[z - start] = weight;
                    weightSum += weight;
                }

                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1] * weightSum;
                for (size_t z = 0; z < end - start; z++)
                {
                    cumsum += weights[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }
                // select next index
                step = indices[index + start];

                // update walk
                walks[i * walkLen + k] = step;
            }
            else
            {
                // searchsorted
                float cumsum = 0;
                size_t index = 0;
                float draw = draws[k - 1];
                for (size_t z = start; z < end; z++)
                {
                    cumsum += data[z];
                    if (draw > cumsum)
                    {
                        continue;
                    }
                    else
                    {
                        index = z;
                        break;
                    }
                }

                // select next index
                step = indices[index];

                // update walk
                walks[i * walkLen + k] = step;
            }
        }
    }
    PARALLEL_FOR_END();

    return _walks;
}

py::array_t<bool> corruptWalks(py::array_t<uint32_t> _walks, size_t nNodes, float r)
{
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    size_t nWalks = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];
    int nCorruptions = (int)(nWalks * walkLen * r);

    py::array_t<bool> _similarity({nWalks, walkLen});
    py::buffer_info similarityBuf = _similarity.request();
    bool *similarity = (bool *)similarityBuf.ptr;
    for (size_t i = 0; i < nWalks * walkLen; i++)
    {
        similarity[i] = true;
    }

    PARALLEL_FOR_BEGIN(nCorruptions)
    {
        // draw random position on matrix
        size_t x = xorshift128() % nWalks;
        size_t y = xorshift128() % (walkLen - 1) + 1;

        // change step by random node
        uint32_t randomNode = xorshift128() % nNodes;
        walks[x * walkLen + y] = randomNode;
        similarity[x * walkLen + y] = false;
    }
    PARALLEL_FOR_END();

    return _similarity;
}

// def _weighted_corrupt_walks(walks, adj_indptr, adj_indices, n_nodes, weights, p=.1):
py::array_t<bool> weightedCorruptWalks(py::array_t<uint32_t> _walks, py::array_t<uint32_t> _candidates, size_t nNodes, float r)
{
    // get data buffers
    py::buffer_info walksBuf = _walks.request();
    uint32_t *walks = (uint32_t *)walksBuf.ptr;

    py::buffer_info candidatesBuf = _candidates.request();
    uint32_t *candidates = (uint32_t *)candidatesBuf.ptr;

    // get data
    size_t nWalks = walksBuf.shape[0];
    size_t walkLen = walksBuf.shape[1];
    size_t nCandidates = candidatesBuf.shape[0];
    int nCorruptions = (int)(nWalks * walkLen * r);

    py::array_t<bool> _similarity({nWalks, walkLen - 1});
    py::buffer_info similarityBuf = _similarity.request();
    bool *similarity = (bool *)similarityBuf.ptr;
    for (size_t i = 0; i < nWalks * (walkLen - 1); i++)
    {
        similarity[i] = true;
    }

    PARALLEL_FOR_BEGIN(nCorruptions)
    {
        // draw random position on matrix
        size_t x = xorshift128() % nWalks;
        size_t y = xorshift128() % (walkLen - 1) + 1;

        // select random node
        size_t randomNode = candidates[xorshift128() % nCandidates];
        walks[x * walkLen + y] = randomNode;

        // change similarity value based on adjacency matrix
        similarity[x * (walkLen - 1) + y - 1] = 0;
        similarity[x * (walkLen - 1) + y] = 0;
    }
    PARALLEL_FOR_END();

    return _similarity;
}
