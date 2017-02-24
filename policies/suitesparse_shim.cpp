// Implementations for SPQR/KLU/CHOLMOD, i.e. "SuiteSparse" implementation

#include <iostream>
#include <cassert>

#include "suitesparse_shim.hpp"

namespace SuiteSparse {

// definition of specializations

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

// sparse matrix constructors
Shim::sparsemat_t::sparsemat_t( ss_shared_ptr<cholmod_sparse> mat )
    : mat_(mat) {}


Shim::sparsemat_t::sparsemat_t( ss_unique_ptr<cholmod_sparse, cholmod_common> && mat )
    : mat_(std::move(mat)) {}

std::ostream&
operator<< ( std::ostream& os, Shim::sparsemat_t const& mat ) {
    // only works for cout/stdout
    assert( &os == &std::cout );
    cholmod_l_write_sparse( stdout, mat.wrapped().get(), nullptr, nullptr,
                            spqr_common.get() );
    fflush(stdout);
    return os;
}

// definitions for calculation methods

// LU
Shim::lu_t::lu_t(Shim::sparsemat_t const& mat)
    : KS_(make_ss_unique_ptr(
              klu_l_analyze(  mat.wrapped()->nrow,
                              reinterpret_cast<long*>(mat.wrapped()->p),
                              reinterpret_cast<long*>(mat.wrapped()->i),
                              klu_common.get() ),
              klu_common)),
      KN_(make_ss_unique_ptr(
              klu_l_factor(   reinterpret_cast<long*>(mat.wrapped()->p),
                              reinterpret_cast<long*>(mat.wrapped()->i),
                              reinterpret_cast<double*>(mat.wrapped()->x),
                              KS_.get(),
                              klu_common.get()),
              klu_common))
{}

Shim::sparsemat_t
Shim::lu_t::solve(sparsemat_t const& B) const {
    // convert B (right hand side) to a dense matrix
    auto Bdense = make_ss_unique_ptr(
        cholmod_l_sparse_to_dense( B.wrapped().get(), spqr_common.get() ),
        spqr_common);

    klu_l_solve ( KS_.get(),          // Symbolic factorization
                  KN_.get(),          // Numeric
                  B.wrapped()->nrow,
                  B.wrapped()->ncol,
                  reinterpret_cast<double*>(Bdense->x),
                  klu_common.get() );

    // convert to cholmod_sparse
    return make_ss_unique_ptr( cholmod_l_dense_to_sparse( Bdense.get(), 1, spqr_common.get() ),
                               spqr_common);
}

// QR
Shim::qr_t::qr_t( sparsemat_t const & mat ) {
    cholmod_sparse * Q;   // results
    cholmod_sparse * R;
    // This is kind of ugly :( SuiteSparseQR returns two pointers by reference
    // Not clear what happens if it can allocate one but not the other
    assert( SuiteSparseQR<double> ( SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, 3,
                                    mat.wrapped().get(),
                                    &Q, &R, nullptr, spqr_common.get() ) >= 0);

    // Now we can finally take ownership
    Q_ = make_ss_shared_ptr( Q, spqr_common );
    R_ = make_ss_unique_ptr( R, spqr_common );
}

Shim::sparsemat_t
Shim::qr_t::Q() const {
    return Q_;
}

}
