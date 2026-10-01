// C++ competitor kernels measured with the same methodology as
// benches/distance.rs and benches/maxsim.rs: one query against a 256-vector
// working set (ns per score), and MaxSim on the same shapes (µs per score).
//
//   hnswlib  InnerProductSpace / L2Space distance functions (v0.8.0 headers)
//   faiss    fvec_inner_product / fvec_L2sqr from the faiss-cpu wheel
//   eigen    MaxSim as Q * D^T + rowwise max (Eigen 3.4)
//   openblas MaxSim as cblas_sgemm + rowwise max (single thread)
//
// Build and run with cpp/run.sh. Output: TSV lines `suite lib shape value unit`.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <vector>

#include <Eigen/Dense>
#include <cblas.h>
#include "hnswlib/hnswlib.h"

namespace faiss {
float fvec_inner_product(const float* x, const float* y, size_t d);
float fvec_L2sqr(const float* x, const float* y, size_t d);
}  // namespace faiss

extern "C" void openblas_set_num_threads(int);

static volatile float g_sink;

// Median over `samples` of the time per call of `body`, auto-scaling the
// number of calls per sample to ~40 ms.
static double median_ns(const std::function<float()>& body, int samples = 30) {
    using clk = std::chrono::steady_clock;
    long reps = 1;
    for (;;) {
        auto t0 = clk::now();
        float s = 0;
        for (long r = 0; r < reps; r++) s += body();
        g_sink = s;
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        if (ms > 40) break;
        reps *= 2;
    }
    std::vector<double> t;
    for (int i = 0; i < samples; i++) {
        auto t0 = clk::now();
        float s = 0;
        for (long r = 0; r < reps; r++) s += body();
        g_sink = s;
        t.push_back(std::chrono::duration<double, std::nano>(clk::now() - t0).count() / reps);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

static std::vector<float> rand_vec(std::mt19937_64& g, size_t n, bool unit = false) {
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<float> v(n);
    for (auto& x : v) x = u(g);
    if (unit) {
        double s = 0;
        for (float x : v) s += double(x) * x;
        float inv = 1.0f / std::sqrt(float(s));
        for (auto& x : v) x *= inv;
    }
    return v;
}

int main() {
    openblas_set_num_threads(1);
    const int SET = 256;
    const size_t dims[] = {96, 100, 128, 256, 384, 768, 960, 1536};
    for (size_t dim : dims) {
        std::mt19937_64 g(dim);
        auto q = rand_vec(g, dim);
        std::vector<float> set(SET * dim);
        for (int i = 0; i < SET; i++) {
            auto v = rand_vec(g, dim);
            std::memcpy(&set[i * dim], v.data(), dim * sizeof(float));
        }
        hnswlib::InnerProductSpace ip(dim);
        hnswlib::L2Space l2(dim);
        auto ipf = ip.get_dist_func();
        auto l2f = l2.get_dist_func();
        void* ipp = ip.get_dist_func_param();
        void* l2p = l2.get_dist_func_param();
        auto per_set = [&](auto f) {
            return median_ns([&] {
                       float acc = 0;
                       for (int i = 0; i < SET; i++) acc += f(&set[i * dim]);
                       return acc;
                   }) /
                   SET;
        };
        printf("dot_f32\thnswlib\t%zu\t%.3f\tns\n", dim,
               per_set([&](const float* v) { return ipf(q.data(), v, ipp); }));
        printf("dot_f32\tfaiss\t%zu\t%.3f\tns\n", dim,
               per_set([&](const float* v) { return faiss::fvec_inner_product(q.data(), v, dim); }));
        printf("l2sq_f32\thnswlib\t%zu\t%.3f\tns\n", dim,
               per_set([&](const float* v) { return l2f(q.data(), v, l2p); }));
        printf("l2sq_f32\tfaiss\t%zu\t%.3f\tns\n", dim,
               per_set([&](const float* v) { return faiss::fvec_L2sqr(q.data(), v, dim); }));
        fflush(stdout);
    }

    struct Shape { int dim, nd, nq; };
    const Shape shapes[] = {{128, 200, 32}, {128, 100, 32}, {128, 300, 32},
                            {96, 200, 32},  {384, 200, 32}, {128, 200, 8}};
    for (auto s : shapes) {
        std::mt19937_64 g(s.dim * 1000 + s.nd + s.nq);
        std::vector<float> Q(s.nq * s.dim), D(s.nd * s.dim), S(s.nq * s.nd);
        for (int i = 0; i < s.nq; i++) {
            auto v = rand_vec(g, s.dim, true);
            std::memcpy(&Q[i * s.dim], v.data(), s.dim * sizeof(float));
        }
        for (int i = 0; i < s.nd; i++) {
            auto v = rand_vec(g, s.dim, true);
            std::memcpy(&D[i * s.dim], v.data(), s.dim * sizeof(float));
        }
        using RM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
        Eigen::Map<const RM> q(Q.data(), s.nq, s.dim), d(D.data(), s.nd, s.dim);
        RM out(s.nq, s.nd);
        char shape[64];
        snprintf(shape, sizeof shape, "d%d/n%d/q%d", s.dim, s.nd, s.nq);
        double eigen = median_ns([&] {
            out.noalias() = q * d.transpose();
            return out.rowwise().maxCoeff().sum();
        });
        double blas = median_ns([&] {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, s.nq, s.nd, s.dim, 1.0f, Q.data(),
                        s.dim, D.data(), s.dim, 0.0f, S.data(), s.nd);
            float total = 0;
            for (int i = 0; i < s.nq; i++)
                total += *std::max_element(&S[i * s.nd], &S[(i + 1) * s.nd]);
            return total;
        });
        printf("maxsim\teigen\t%s\t%.3f\tus\n", shape, eigen / 1000);
        printf("maxsim\topenblas\t%s\t%.3f\tus\n", shape, blas / 1000);
        fflush(stdout);
    }
}
