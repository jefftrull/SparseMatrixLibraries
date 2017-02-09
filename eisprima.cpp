// Using Eigen to compute a QR decomposition

#include <vector>
#include <tuple>
#include <cassert>
#include <iostream>

#include <Eigen/SparseQR>
#include <Eigen/SparseLU>

int main ()
{
    using namespace std;
    using namespace Eigen;

    vector<Triplet<double>> Gentries{
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

    vector<Triplet<double>> Bentries{
        {12, 0, -1},
        {13, 1, -1},
        {14, 2, -1}};

    // build sparse matrices from triplet lists
    SparseMatrix<double> G(15, 15);
    G.setFromTriplets(Gentries.begin(), Gentries.end());
    SparseMatrix<double> B(15, 3);
    B.setFromTriplets(Bentries.begin(), Bentries.end());

    // calculate A = G^-1*B with Sparse LU
    SparseLU<SparseMatrix<double>, COLAMDOrdering<int>> LU(G);
    assert(LU.info() == Success);
    SparseMatrix<double> A = LU.solve(B);

    // run QR on A
    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> QR(A);
    assert(QR.info() == Success);

    IOFormat OctaveFmt(FullPrecision, 0, ", ", ";\n", "", "", "[", "]");

    // generate Q:
    // store the Q matrix as dense
    MatrixXd Q = QR.matrixQ() * MatrixXd::Identity(15, QR.rank());

    std::cout << "Q=" << Q.format(OctaveFmt) << "\n";

}
