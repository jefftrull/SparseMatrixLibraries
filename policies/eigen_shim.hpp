// describe the type requirements our code has from a sparse library
#include "sparselib_concept.hpp"

// Implement those requirements using Eigen
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/SparseLU>

struct EigenShim {
    using value_t = double;
    using triplet_t = Eigen::Triplet<value_t>;

    template<typename Value>
    struct sparse_wrapper_t {
        using wrapped_t = Eigen::SparseMatrix<Value>;
        using index_t = typename wrapped_t::StorageIndex;
        template<typename Iter>     // or use ForwardIterator concept
        sparse_wrapper_t(index_t rows, index_t cols, Iter a, Iter b) : mat_(rows, cols) {
            mat_.setFromTriplets(a, b);
        }
        sparse_wrapper_t(wrapped_t mat) : mat_(std::move(mat)) {}

        // define the product of two sparse wrappers
        friend sparse_wrapper_t operator*(sparse_wrapper_t const& a, sparse_wrapper_t const& b) {
            return Eigen::SparseMatrix<value_t>(a.wrapped() * b.wrapped());
        }

        friend std::ostream & operator<<(std::ostream& os, sparse_wrapper_t const& m) {
            using namespace Eigen;
            IOFormat OctaveFmt(FullPrecision, 0, ", ", ";\n", "", "", "[", "]");

            Matrix<value_t, Dynamic, Dynamic> dense = m.wrapped();   // convert to dense

            os << dense.format(OctaveFmt);
            return os;
        }

        wrapped_t const & wrapped() const { return mat_; }

    private:

        wrapped_t mat_;
    };

    using sparsemat_t = sparse_wrapper_t<value_t>;
    using index_t = sparsemat_t::index_t;

    template<typename Value, typename Index>
    struct lu_wrapper_t {
        using wrapped_t = Eigen::SparseLU<Eigen::SparseMatrix<Value>, Eigen::COLAMDOrdering<Index>>;

        lu_wrapper_t( sparsemat_t const & mat ) : lu_(mat.wrapped()) {
            assert(lu_.info() == Eigen::Success);
        }

        sparse_wrapper_t<Value> solve( sparsemat_t const & rhs ) const {
            return sparse_wrapper_t<Value>(lu_.solve(rhs.wrapped()));
        }

    private:
        wrapped_t lu_;
    };

    template<typename Value, typename Index>
    struct qr_wrapper_t {
        using wrapped_t = Eigen::SparseQR<Eigen::SparseMatrix<Value>, Eigen::COLAMDOrdering<Index>>;

        qr_wrapper_t( sparsemat_t const & mat ) : qr_(mat.wrapped()) {}

        sparsemat_t Q() const {
            using namespace Eigen;
            // Sadly Eigen cannot directly return the Q as a sparse matrix
            // What it *can* do is multiply times a dense matrix
            Matrix<value_t, Dynamic, Dynamic> identity(qr_.rows(), qr_.rank());
            identity.setIdentity();

            Matrix<value_t, Dynamic, Dynamic> result = qr_.matrixQ() * identity;

            // finally, convert to sparse
            return SparseMatrix<value_t>(result.sparseView());
        }

    private:
        wrapped_t qr_;
    };

    using lu_t = lu_wrapper_t<value_t, index_t>;
    using qr_t = qr_wrapper_t<value_t, index_t>;


};

