// Using Eigen to compute a QR decomposition

#include <vector>
#include <tuple>
#include <cassert>
#include <iostream>

#include <Eigen/SparseQR>

int main ()
{
    using namespace std;
    using namespace Eigen;

    vector<Triplet<double> > entries{
        {0, 0, 1.},
        {1, 0, 1.},
        {2, 0, 1.},
        {3, 0, 1.},
        {4, 0, 1.},
        {5, 0, 1.},
        {6, 1, 1.},

        {7, 1, 0.952380952380952},
        {7, 2, 0.0476190476190476},

        {8, 1, 0.714285714285714},
        {8, 2, 0.285714285714286},

        {9, 1, 0.476190476190476},
        {9, 2, 0.523809523809524},

        {10, 1, 0.238095238095238},
        {10, 2, 0.761904761904762},

        {11, 2, 1.},

//        {12, 0, -8.13151629364128e-19},

        {13, 1, -0.000476190476190477},
        {13, 2, 0.000476190476190476},

        {14, 1, 0.000476190476190476},
        {14, 2, -0.000476190476190476}};

    SparseMatrix<double> A(15, 3);
    A.setFromTriplets(entries.begin(), entries.end());

    // run QR
    SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> QR(A);
    assert(QR.info() == Success);

    IOFormat OctaveFmt(FullPrecision, 0, ", ", ";\n", "", "", "[", "]");

    // generate Q:
    // store the Q matrix as dense
    MatrixXd Q = QR.matrixQ() * MatrixXd::Identity(15, QR.rank());

    std::cout << "Q=" << Q.format(OctaveFmt) << "\n";

}
