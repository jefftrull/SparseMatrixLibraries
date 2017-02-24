// SuiteSparse policy definition

#include <memory>

#include <SuiteSparseQR.hpp>
#include <klu.h>

namespace SuiteSparse {

// utility classes

// wrapping SuiteSparse memory allocation
template<typename Common>
struct ss_deleter {
    ss_deleter() : cc_(nullptr) {}

    ss_deleter(Common * cc) : cc_(cc) {}

    void operator()(cholmod_triplet *& p) const {
        cholmod_l_free_triplet(&p, cc_);
    }
    void operator()(cholmod_sparse *& p) const {
        cholmod_l_free_sparse(&p, cc_);
    }
    void operator()(cholmod_dense *& p) const {
        cholmod_l_free_dense(&p, cc_);
    }
    void operator()(klu_l_symbolic *& p) const {
        klu_l_free_symbolic (&p, cc_);
    }
    void operator()(klu_l_numeric *& p) const {
        klu_l_free_numeric (&p, cc_);
    }

private:
    Common * cc_;

};

template<typename T, typename Common>
using ss_unique_ptr = std::unique_ptr<T, ss_deleter<Common>>;

template<typename T, typename Common>
ss_unique_ptr<T, typename Common::wrapped_t>
make_ss_unique_ptr( T* p, Common & c ) {
    return ss_unique_ptr<T, typename Common::wrapped_t>(p, c.get());
}

template<typename T>
using ss_shared_ptr = std::shared_ptr<T>;

template<typename T, typename Common>
ss_shared_ptr<T>
make_ss_shared_ptr( T* p, Common & c ) {
    return ss_shared_ptr<T>(p, ss_deleter<typename Common::wrapped_t>(c.get()));
}

// Define a wrapper for SuiteSparse "common" objects
// takes care of calling start and finish cleanly,
// and supplies a deleter (which needs a common reference)

template<typename Common>
struct common_wrapper {

    using wrapped_t = Common;

    common_wrapper() {}
    ~common_wrapper() {}

    // no copies, no assignments
    common_wrapper(common_wrapper const& other) = delete;
    common_wrapper & operator=(common_wrapper const&) = delete;

    wrapped_t * get() {
        return &common_;
    }

    ss_deleter<Common> deleter() {
        return ss_deleter<Common>(&common_);
    }

    private:

    Common common_;
};

// specialize constructor/destructor for the different kinds of common objects
// to call the appropriate C routine
template<>
common_wrapper<cholmod_common>::common_wrapper();
template<>
common_wrapper<cholmod_common>::~common_wrapper();
template<>
common_wrapper<klu_l_common>::common_wrapper();


// SPQR and KLU both require the use of a "common" object that gets passed in
// to each method call.  It seems like it the options for handling this are:
// 1) it is static and this code "knows" about it
// 2) require callers to pass it in - a change to the Concept and unnecessary
//    for other libraries
// 3) change the API so everything is a factory function from a single object
//    (also a Concept change)
// None of these options make me happy.  For now I'm just going to do item 1)
// since this is an exercise anyway.

static common_wrapper<klu_l_common> klu_common;
static common_wrapper<cholmod_common> spqr_common;

struct Shim {
    using value_t = double;
    using index_t = long;    // Compatible with "_l" functions

    struct triplet_t {
        index_t row;
        index_t col;
        value_t value;
    };

    struct sparsemat_t {
        template<typename Iter>
        sparsemat_t( index_t rows, index_t cols,
                     Iter first, Iter last ) {
            // load into "triplet matrix"
            auto Gct = make_ss_unique_ptr(
                cholmod_l_allocate_triplet(
                    rows, cols, std::distance(first, last),
                    0,  // stype: both upper and lower are stored
                    CHOLMOD_REAL,
                    spqr_common.get()),
                spqr_common);
            for ( Iter it = first; it < last; ++it) {
                index_t idx = std::distance(first, it);
                reinterpret_cast<long *>(Gct->i)[idx]   = it->row;
                reinterpret_cast<long *>(Gct->j)[idx]   = it->col;
                reinterpret_cast<double *>(Gct->x)[idx] = it->value;
            }
            Gct->nnz = std::distance(first, last);

            // convert triplet matrix to sparse
            mat_ = make_ss_unique_ptr(
                cholmod_l_triplet_to_sparse(Gct.get(), std::distance(first, last), spqr_common.get()),
                spqr_common);
        }

        sparsemat_t( ss_shared_ptr<cholmod_sparse> );

        sparsemat_t( ss_unique_ptr<cholmod_sparse, cholmod_common> && );

        friend sparsemat_t operator*(sparsemat_t const& a, sparsemat_t const& b);
        friend std::ostream& operator<<(std::ostream& os, sparsemat_t const & m);

        ss_shared_ptr<cholmod_sparse> wrapped() const {
            return mat_;
        }

    private:
        ss_shared_ptr<cholmod_sparse> mat_;

    };

    struct lu_t {
        lu_t( sparsemat_t const & mat );

        sparsemat_t solve(sparsemat_t const& rhs) const;

    private:
        ss_unique_ptr<klu_l_symbolic, klu_l_common> KS_;
        ss_unique_ptr<klu_l_numeric, klu_l_common>  KN_;

    };

    struct qr_t {
        qr_t( sparsemat_t const & mat );

        sparsemat_t Q() const;

    private:
        ss_shared_ptr<cholmod_sparse> Q_;
        ss_unique_ptr<cholmod_sparse, cholmod_common>  R_;
    };

};


}
