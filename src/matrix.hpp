#include <stdlib.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <random>
#include <emmintrin.h>


namespace py = pybind11;


namespace matrix {
    int unravel(int x, int y, int width){
        return y + x * width;
    }

    template<typename T>
    class Vector
    {
        private:
        public:
            int size;
            T* X;

            Vector(py::array_t<T>& X_){
                py::buffer_info XBuf = X_.request();
                X = (T *)XBuf.ptr;
                size = XBuf.shape[0];
            }

            float get(int i)
            {
                return X[i];
            }

            void set(int i, T value)
            {
                X[i] = value;
            }
    };


    template<typename T>
    class Matrix
    {
        private:
            T* X;
        public:
            size_t rows;
            size_t dim;

            Matrix(py::array_t<T>& X_){
                py::buffer_info XBuf = X_.request();
                // X = (T *)XBuf.ptr;
                X = static_cast<T *>(XBuf.ptr);
                rows = XBuf.shape[0];
                dim = XBuf.shape[1];
            }

            float pair_quadrance(size_t k,  size_t j){
                size_t aligned = dim - (dim % 4);
                float quad = 0;
                for (size_t z = aligned; z < dim; z++)
                {
                    const float num = X[k * dim + z] - X[j * dim + z];
                    quad += num * num;
                }

                float* a = &X[k * dim];
                float* b = &X[j * dim];
                __m128 quadrance = _mm_setzero_ps();
                for (size_t i = 0; i < aligned; i += 4){
                    const __m128 x = _mm_loadu_ps(a);
                    const __m128 y = _mm_loadu_ps(b);
                    const __m128 a_minus_b = _mm_sub_ps(x, y);
                    const __m128 a_minus_b_sq = _mm_mul_ps(a_minus_b, a_minus_b);
                    quadrance = _mm_add_ps(quadrance, a_minus_b_sq);
                    b += 4;
                    a += 4;
                }
                
                // compute sum
                __m128 shuf = _mm_shuffle_ps(quadrance, quadrance, _MM_SHUFFLE(2, 3, 0, 1));
                __m128 sums = _mm_add_ps(quadrance, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                quad += _mm_cvtss_f32(sums);

                return quad;
            }

            float quadrance_from(size_t k,  float* b){
                float* a = &X[k * dim];

                size_t aligned = dim - (dim % 4);

                float quad = 0;
                for (size_t z = aligned; z < dim; z++)
                {
                    const float num = X[k * dim + z] - b[z];
                    quad += num * num;
                }

                __m128 quadrance = _mm_setzero_ps();
                for (size_t i = 0; i < aligned; i += 4){
                    const __m128 x = _mm_loadu_ps(a);
                    const __m128 y = _mm_loadu_ps(b);
                    const __m128 a_minus_b = _mm_sub_ps(x, y);
                    const __m128 a_minus_b_sq = _mm_mul_ps(a_minus_b, a_minus_b);
                    quadrance = _mm_add_ps(quadrance, a_minus_b_sq);
                    b += 4;
                    a += 4;
                }
                
                // compute sum
                __m128 shuf = _mm_shuffle_ps(quadrance, quadrance, _MM_SHUFFLE(2, 3, 0, 1));
                __m128 sums = _mm_add_ps(quadrance, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                quad += _mm_cvtss_f32(sums);

                return quad;
            }

            float get(int i, int j)
            {
                return X[unravel(i, j, dim)];
            }

            void set(int i, int j, T value){
                X[unravel(i, j, dim)] = value;
            }

            void add(int i, int j, T value){
                X[unravel(i, j, dim)] += value;
            }
    };
}