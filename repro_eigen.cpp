// Reproducing an uncommon QR failure found in random testing (sparseqr_1)

#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SparseQR>

int main() {

    using namespace Eigen;

    using Scalar = double;
    using DenseMatrix = Matrix<Scalar, 4, 5>;

    // FullPrecision version
    std::vector<Triplet<Scalar>> tlist{
        {0, 0, 10.8750122285634},
        {1, 0, -0.397596622536702},
        {1, 1, 12.1402677156793},
        {1, 4, 0.031725381050131},
        {2, 0, -0.851737443288666},
        {2, 1, -0.0269339247732115},
        {2, 2, 11.3112600706892},
        {2, 3, 0.013059221679838},
        {3, 0, -0.676105666754816},
        {3, 2, 0.138751648896724},
        {3, 3, 8.57744542262346}};

    DenseMatrix A = DenseMatrix::Zero();
    for (auto const & trpl : tlist) {
        A.coeffRef(trpl.row(), trpl.col()) = trpl.value();
    }

    Matrix<Scalar, 4, 1> b;
    // FullPrecision
    b << 10.3612221487294, -2.27836032745297, -10.3179355124181, -7.49344052343591;

    auto qr = A.colPivHouseholderQr();

    Matrix<Scalar, 5, 1> x = qr.solve(b);

    IOFormat OctaveFmt(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");

    std::cerr << "A=\n" << A.format(OctaveFmt) << "\n";
    std::cerr << "b=\n" << b.format(OctaveFmt) << "\n";
    std::cerr << "x=\n" << x.format(OctaveFmt) << "\n";

    // ^dense results^ are fine

    SparseMatrix<Scalar> sA(4, 5); sA.setFromTriplets(tlist.begin(), tlist.end());

    SparseQR<SparseMatrix<Scalar>, COLAMDOrdering<int>> sqr(sA);
    // no indication of a problem here
    if (sqr.info() != Success) {
        std::cerr << "decomp failed!\n";
    }

    Matrix<Scalar, Dynamic, Dynamic> sx = sqr.solve(b);
    // or here
    if (sqr.info() != Success) {
        std::cerr << "solve failed!\n";
    }

    // but this result is badly off:
    std::cerr << "sx=\n" << sx.format(OctaveFmt) << "\n";

    // specifically because we cannot recover b:
    Matrix<Scalar, Dynamic, Dynamic> sbrecover = sA * sx;
    std::cerr << "recovered b=\n" << sbrecover.format(OctaveFmt) << "\n";
       

    // debug
    std::cerr << "dense Q=\n" << Matrix<Scalar, Dynamic, Dynamic>(qr.matrixQ()).format(OctaveFmt) << "\n";
    std::cerr << "sparse Q=\n" << Matrix<Scalar, Dynamic, Dynamic>(sqr.matrixQ()).format(OctaveFmt) << "\n";

    // can we recover the original from sparse?
    std::cerr << "recovering original A from sparse:\n" << (Matrix<Scalar, Dynamic, Dynamic>(sqr.matrixQ()) * sqr.matrixR().template triangularView<Upper>()*sqr.colsPermutation().transpose()).format(OctaveFmt) << "\n";
    // it looks pretty OK actually

    // summary for debugging
    std::cerr << "rank is " << sqr.rank() << " and permutation is:\n";
    auto indices = sqr.colsPermutation().indices();
    std::cerr << "permutation indices:\n";
    std::cerr << indices.format(OctaveFmt) << "\n";
    std::cerr << "permutation matrix:\n";
    std::cerr << Matrix<Scalar, Dynamic, Dynamic>(sqr.colsPermutation()).format(OctaveFmt) << "\n";

/*    for (auto const & i : indices) {
        std::cerr << i << " ";
    }
    std::cerr << "\n";
*/

}

