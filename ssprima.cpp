// Illustration of (direct) use of SuiteSparse and SQPR
// based on example from SuiteSparse docs

#include <vector>
#include <tuple>
#include <cassert>
#include <memory>

#include <SuiteSparseQR.hpp>
#include <klu.h>

// Use std::unique_ptr plus a custom deleter to clean up the C-style manual memory management

template<typename Common>
struct ss_deleter {
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
using ss_ptr = std::unique_ptr<T, ss_deleter<Common>>;

template<typename T, typename Common>
ss_ptr<T, typename Common::wrapped_t>
make_ss_ptr( T* p, Common & c ) {
    return std::move(ss_ptr<T, typename Common::wrapped_t>(p, c.get()));
}

// take care of calling start and finish cleanly
// also supply deleters as needed

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

template<>
common_wrapper<cholmod_common>::common_wrapper() {
    cholmod_l_start(&common_);
}

template<>
common_wrapper<cholmod_common>::~common_wrapper() {
    cholmod_l_finish(&common_);
}

template<>
common_wrapper<klu_l_common>::common_wrapper() {
    klu_l_defaults(&common_);
}

struct triplet {
    int row;
    int col;
    double value;
};

int main ()
{
    using namespace std;

    common_wrapper<cholmod_common> ccommon;
    // start CHOLMOD
    cholmod_common * cc = ccommon.get() ;

    vector<triplet> Gentries{
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

    vector<triplet> Bentries{
        {12, 0, -1},
        {13, 1, -1},
        {14, 2, -1}};

    // now load into Cholmod "triplet matrices"

    auto Gct = make_ss_ptr(
        cholmod_l_allocate_triplet(
            15, 15, Gentries.size(),
            0,  // stype: both upper and lower are stored
            CHOLMOD_REAL,
            cc),
        ccommon);
    for ( size_t i = 0; i < Gentries.size(); ++i) {
        reinterpret_cast<long *>(Gct->i)[i] = Gentries[i].row;
        reinterpret_cast<long *>(Gct->j)[i] = Gentries[i].col;
        reinterpret_cast<double *>(Gct->x)[i] = Gentries[i].value;
    }
    Gct->nnz = Gentries.size();

    // convert triplet matrix to sparse
    auto G = make_ss_ptr(
        cholmod_l_triplet_to_sparse(Gct.get(), Gentries.size(), cc),
        ccommon);

    // RHS needs to be dense due to API, so build B that way:

    // initialize A (the eventual result) with the contents of B
    auto Adense = make_ss_ptr(
        cholmod_l_zeros( 15, 3, CHOLMOD_REAL, cc ),
        ccommon);
    for ( size_t i = 0; i < Bentries.size(); ++i) {
        reinterpret_cast<double*>(Adense->x)[15*Bentries[i].col+Bentries[i].row] = Bentries[i].value;
    }

    // calculate G^-1*B via LU with the KLU package from SuiteSparse
    // It claims to be "well suited for circuit simulation", which this is
    common_wrapper<klu_l_common> kcommon;
    auto KS = make_ss_ptr(klu_l_analyze( 15,
                                         reinterpret_cast<long*>(G->p),
                                         reinterpret_cast<long*>(G->i),
                                         kcommon.get() ),
                          kcommon);
    auto KN = make_ss_ptr(klu_l_factor(   reinterpret_cast<long*>(G->p),
                                          reinterpret_cast<long*>(G->i),
                                          reinterpret_cast<double*>(G->x),
                                          KS.get(),
                                          kcommon.get() ),
                          kcommon);

    klu_l_solve ( KS.get(),          // Symbolic factorization
                  KN.get(),          // Numeric
                  15,
                  3,
                  reinterpret_cast<double*>(Adense->x),
                  kcommon.get() );

    // convert to cholmod_sparse
    auto A = make_ss_ptr(
        cholmod_l_dense_to_sparse( Adense.get(), 1, cc ),
        ccommon);

    // run QR
    cholmod_sparse * Q;   // outputs
    cholmod_sparse * R;
    assert( SuiteSparseQR<double> ( SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, 3, A.get(),
                                    &Q, &R, nullptr, cc ) >= 0);

    // print out Q matrix
    cholmod_l_write_sparse( stdout, Q , NULL, NULL, cc );

    // free everything and finish CHOLMOD

    cholmod_l_free_sparse (&Q, cc) ;
    cholmod_l_free_sparse (&R, cc) ;

    return (0) ;
}
