// Implementation of CSparseShim
// some very small functions are implemented in the header...

#include "csparse_shim.hpp"

// sparse matrix entry iterator
void
CSparseShim::sparse_entry_iterator::increment() {
    entry_.row++;
    advance_to_valid();
}

bool
CSparseShim::sparse_entry_iterator::equal( sparse_entry_iterator const& other ) const {
    if ( !mat_ && !other.mat_ ) {
        // both end iterators
        return true;
    }

    return (mat_ && other.mat_ &&
            (entry_.row == other.entry_.row) &&
            (entry_.col == other.entry_.col));
}

void
CSparseShim::sparse_entry_iterator::advance_to_valid() {
    if (!mat_) {
        // end iterator; always valid
        return;
    }

    // look for a valid entry
    // note a couple of optimization opportunities here:
    // - if row indices within column are sorted we can binary search
    // - if nonzero entry order is unimportant, can emit entries in the
    //   order we found them
    for ( ; entry_.col < mat_->n; entry_.col++ ) {
        index_t col_start = mat_->p[entry_.col];
        index_t col_end   = mat_->p[entry_.col+1];
        for ( ; entry_.row < mat_->m ; entry_.row++ ) {
            // can we find this row within the indices?
            index_t * loc = std::find( mat_->i + col_start,
                                       mat_->i + col_end,
                                       entry_.row );
            if ( loc < mat_->i + col_end ) {
                entry_.value = mat_->x[std::distance(mat_->i, loc)];
                return;
            }
        }
        entry_.row = 0;
    }

    // if we didn't return, we have run off the end of the matrix
    mat_.reset();  // indicate end
}

// solvers

CSparseShim::sparsemat_t
CSparseShim::lu_t::solve(sparsemat_t const& rhs) const {
    using namespace std;
    // turn RHS into a dense (zero) matrix
    vector<value_t> rhs_dense( rhs.rows() * rhs.cols() );
    fill( rhs_dense.begin(), rhs_dense.end(), value_t{0} );

    // Loop over nonzero values and insert
    for ( auto it = rhs.nonzero_begin(); it != rhs.nonzero_end(); ++it ) {
        rhs_dense[it->col * rhs.rows() + it->row] = it->value;
    }

    // solve one column at a time
    vector<value_t> workspace(max(rhs.cols(), rhs.rows()));
    for ( index_t col = 0; col < rhs.cols(); ++col ) {
        cs_ipvec  ( numeric_->pinv, &rhs_dense[rhs.rows()*col],
                    workspace.data(), rhs.rows() );
        cs_lsolve ( numeric_->L, workspace.data() ) ;
        cs_usolve ( numeric_->U, workspace.data() ) ;
        cs_ipvec  ( symbolic_->q, workspace.data(),
                    &rhs_dense[rhs.rows()*col], rhs.rows() );
    }                

    // produce a sparse matrix from the dense result
    return dense_to_sparse(rhs_dense, rhs.rows(), rhs.cols());

}            

CSparseShim::sparsemat_t
CSparseShim::qr_t::Q() const {

    // reverse the solve permutation
    auto P = cs_unique_ptr<index_t>(cs_pinv(symbolic_->pinv, rows_));

    // allocate workspace
    std::vector<value_t> x(symbolic_->m2, value_t{0});

    cs* V = numeric_->L;

    // create a *dense* identity matrix of the right size
    index_t rank = V->n;
    std::vector<value_t> Q(rows_ * rank);
    for ( index_t idx = 0; idx < std::min(rows_, rank); idx++ ) {
        Q[idx * rows_ + idx] = value_t(1);
    }

    // proceed one column at a time (see cs_qrsol.c)
    for ( index_t j = 0; j < rank; j++) {
        double * col = &Q[rows_*j];
        // make x a copy of the jth column of I
        std::copy(col, col+rows_, x.data());

        // apply the Householder vectors that comprise Q
        for (index_t k = j; k >= 0; k--) {
            cs_happly( V, k, numeric_->B[k], x.data() );
        }

        // apply the row permutation
        cs_ipvec( P.get(), x.data(), col, rows_ );
    }

    return dense_to_sparse(Q, rows_, rank);

}

// utility functions

CSparseShim::sparsemat_t
CSparseShim::dense_to_sparse( std::vector<value_t> const& d, index_t rows, index_t cols ) {

    // we need to count the nonzeros to allocate
    index_t result_nz = count_if(d.begin(), d.end(),
                                 [](value_t x) { return x != value_t(0); });

    cs_unique_ptr<cs> result(cs_spalloc(rows, cols, result_nz, 1, 1));

    for ( index_t j = 0; j < cols; ++j) {
        for ( index_t i = 0; i < rows; ++i) {
            if ( d[rows*j+i] != value_t(0) ) {
                assert(cs_entry(result.get(), i, j, d[rows*j+i]));
            }
        }
    }

    return sparsemat_t(make_cs_shared_ptr(cs_compress(result.get())));

}

std::ostream& operator<<(std::ostream& os, CSparseShim::sparsemat_t const & m) {
    assert(&os == &std::cout);   // because cs_print only does stdout
    cs_print( m.wrapped().get(), 0 );
    fflush(stdout);
}
