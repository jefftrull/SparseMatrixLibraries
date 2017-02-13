#include <vector>
#include <iostream>

#ifdef USE_EIGEN
#include "eigen_shim.hpp"
using sparse_lib_t = EigenShim;
#endif

// generic code that uses the Concept

template<SparseLibrary L>
requires ConstructibleFrom<typename L::sparsemat_t, typename L::index_t, typename L::index_t,
                           typename std::vector<typename L::triplet_t>::const_iterator,
                           typename std::vector<typename L::triplet_t>::const_iterator>
void startPrima() {
    // run the first few steps of Prima using our SparseLibrary
    using namespace std;
    vector<typename L::triplet_t> Gentries{
        {0, 0, 0.01},
        {0, 1, -0.01},
        {0, 12, 1},
        {1, 0, -0.01},
        {1, 1, 0.012},
        {1, 2, -0.002},
        {2, 1, -0.002},
        {2, 2, 0.004},
        {2, 3, -0.002},
        {3, 2, -0.002},
        {3, 3, 0.004},
        {3, 4, -0.002},
        {4, 3, -0.002},
        {4, 4, 0.004},
        {4, 5, -0.002},
        {5, 4, -0.002},
        {5, 5, 0.002},
        {6, 6, 0.01},
        {6, 7, -0.01},
        {6, 13, 1},
        {7, 6, -0.01},
        {7, 7, 0.012},
        {7, 8, -0.002},
        {8, 7, -0.002},
        {8, 8, 0.004},
        {8, 9, -0.002},
        {9, 8, -0.002},
        {9, 9, 0.004},
        {9, 10, -0.002},
        {10, 9, -0.002},
        {10, 10, 0.004},
        {10, 11, -0.002},
        {11, 10, -0.002},
        {11, 11, 0.002},
        {11, 14, 1},
        {12, 0, -1},
        {13, 6, -1},
        {14, 11, -1}
    };

    vector<typename L::triplet_t> Bentries{
        {12, 0, -1},
        {13, 1, -1},
        {14, 2, -1}};

    // build sparse matrices from triplet lists
    typename L::sparsemat_t G(15, 15, begin(Gentries), end(Gentries));
    typename L::sparsemat_t B(15, 3, begin(Bentries), end(Bentries));

    // calculate A = G^-1*B with Sparse LU
    typename L::lu_t LU(G);
    auto A = LU.solve(B);

    // run QR on A
    typename L::qr_t QR(A);
    auto Q = QR.Q();

    // display result
    std::cout << "Q=\n" << Q << "\n";

}    



int main() {
    startPrima<sparse_lib_t>();

/*
    std::vector<EigenShim::triplet_t> triplets;
    EigenShim::sparsemat_t s(15, 15, triplets.begin(), triplets.end());
    EigenShim::lu_t lu(s);
    EigenShim::lu_t bar = lu;
*/
}
