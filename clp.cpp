#include <cstring>
#include <limits>
#include <random>
#include <cstdio>
#include <cmath>
#include <fstream>

using prec_t=double;


template<typename T, const int N>
struct Vector {
    T vec[N];
    Vector() {
        memset(vec, 0, sizeof vec);
    }
    T& operator[] (const int &i) {
        return vec[i];
    }
    const T& operator[] (const int &i) const {
        return vec[i];
    }
    Vector<T, N>& operator= (const Vector<T, N> &other) {
        memcpy(vec, other.vec, sizeof(T) * N);
        return *this;
    }
};


template<typename T, const int N>
struct Matrix {
    Vector<T, N> mat[N];
    Vector<T, N>& operator[] (const int &i) {
        return mat[i];
    }
    const Vector<T, N>& operator[] (const int &i) const {
        return mat[i];
    }
};

template<typename T, const int N>
Vector<T, N> mult(const Vector<T, N> &a, const Matrix<T, N> &b) {
    Vector<T, N> c;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            c[i] += a[j] * b[j][i];
        }
    }
    return c;
}

prec_t sign(const prec_t &x) {
    return prec_t(x > 0 ? 1 : -1);
}

template<const int N>
Vector<prec_t, N> compute_clp(Vector<prec_t, N> r, Matrix<prec_t, N> G) {
    prec_t C = std::numeric_limits<prec_t>::infinity();
    int i = N;
    Vector<int, N> d;
    for (int j = 0; j < N; ++j) {
        d[j] = N - 1;
    }
    Matrix<prec_t, N> F;
    F[N - 1] = r;
    Vector<prec_t, N + 1> lam;
    Vector<prec_t, N> u, hat_u;
    Vector<prec_t, N> p;
    Vector<prec_t, N> delta;

    while (true) {
        do {
            if (i > 0) {
                --i;
                for (int j = d[i]; j > i; --j) {
                    F[j - 1][i] = F[j][i] - u[j] * G[j][i];
                }
                p[i] = F[i][i] / G[i][i];
                u[i] = round(p[i]);
                // printf("%f\n", p[i] - u[i]);
                prec_t y = (p[i] - u[i]) * G[i][i];
                delta[i] = sign(y);
                lam[i] = lam[i + 1] + y * y;
            }
            else {
                hat_u = u;
                C = lam[0];
                // printf("%f\n", C);
            }
        } while (lam[i] < C);
        int m = i;
        do {
            if (i < N - 1) {
                ++i;
                u[i] += delta[i];
                delta[i] = -delta[i] - sign(delta[i]);
                prec_t y = (p[i] - u[i]) * G[i][i];
                lam[i] = lam[i + 1] + y * y;
            }
            else {
                return hat_u;
            }
        } while (lam[i] >= C);
        for (int j = m; j < i; ++j) {
            if (d[j] < i) {
                d[j] = i;
            }
            else {
                break;
            }
        }
    }
}

template<const int N>
prec_t compute_nsm(Matrix<prec_t, N> B, std::mt19937 gen) {
    prec_t V = 1;
    for (int i = 0; i < N; ++i) {
        V *= B[i][i];
    }
    std::uniform_real_distribution<prec_t> dist(0, 1);
    prec_t norms = 0;
    const int T = 200000;
    for (int i = 0; i < T; ++i) {
        Vector<prec_t, N> z;
        for (int j = 0; j < N; ++j) {
            z[j] = dist(gen);
        }
        Vector<prec_t, N> u = mult(z, B);
        Vector<prec_t, N> pz = compute_clp(u, B);
        Vector<prec_t, N> y;
        for (int j = 0; j < N; ++j) {
            y[j] = z[j] - pz[j];
        }
        Vector<prec_t, N> e = mult(y, B);
        prec_t thisnorm = 0;
        for (int j = 0; j < N; ++j) {
            thisnorm += e[j] * e[j];
        }
        norms += thisnorm / T;
    }
    return norms / (N * pow(V, 2.0 / N));
}

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::ifstream f("best.npy.txt");
    Matrix<prec_t, 32> G;
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            f >> G[i][j];
        }
    }
    printf("%f\n", compute_nsm(G, gen));
    return 0;
}