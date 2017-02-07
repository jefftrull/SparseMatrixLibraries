// Illustration of (direct) use of SuiteSparse and SQPR
// based on example from SuiteSparse docs

#include <vector>
#include <tuple>
#include <cassert>

#include <SuiteSparseQR.hpp>
#include <klu.h>

struct triplet {
    int row;
    int col;
    double value;
};

int main ()
{
    using namespace std;

    cholmod_common Common, *cc ;
    // start CHOLMOD
    cc = &Common ;
    cholmod_l_start (cc) ;

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

    cholmod_triplet * Gct = cholmod_l_allocate_triplet(
        15, 15, Gentries.size(),
        0,  // stype: both upper and lower are stored
        CHOLMOD_REAL,
        cc);
    for ( size_t i = 0; i < Gentries.size(); ++i) {
        reinterpret_cast<long *>(Gct->i)[i] = Gentries[i].row;
        reinterpret_cast<long *>(Gct->j)[i] = Gentries[i].col;
        reinterpret_cast<double *>(Gct->x)[i] = Gentries[i].value;
    }
    Gct->nnz = Gentries.size();

    // convert triplet matrix to sparse
    cholmod_sparse * G = cholmod_l_triplet_to_sparse(Gct, Gentries.size(), cc);

    // RHS needs to be dense due to API, so build B that way:

    // initialize A (the eventual result) with the contents of B
    cholmod_dense * Adense = cholmod_l_zeros( 15, 3, CHOLMOD_REAL, cc );
    for ( size_t i = 0; i < Bentries.size(); ++i) {
        reinterpret_cast<double*>(Adense->x)[15*Bentries[i].col+Bentries[i].row] = Bentries[i].value;
    }

    // calculate G^-1*B via LU with the KLU package from SuiteSparse
    // It claims to be "well suited for circuit simulation", which this is
    klu_l_common kcommon;
    klu_l_defaults( &kcommon );
    klu_l_symbolic * KS = klu_l_analyze( 15,
                                         reinterpret_cast<long*>(G->p),
                                         reinterpret_cast<long*>(G->i),
                                         &kcommon );
    klu_l_numeric  * KN = klu_l_factor(  reinterpret_cast<long*>(G->p),
                                         reinterpret_cast<long*>(G->i),
                                         reinterpret_cast<double*>(G->x),
                                         KS,
                                         &kcommon );
    klu_l_solve ( KS,          // Symbolic factorization
                  KN,          // Numeric
                  15,
                  3,
                  reinterpret_cast<double*>(Adense->x),
                  &kcommon );

    // convert to cholmod_sparse
    cholmod_sparse * A = cholmod_l_dense_to_sparse( Adense, 1, cc );

    // run QR
    cholmod_sparse * Q;   // outputs
    cholmod_sparse * R;
    assert( SuiteSparseQR<double> ( SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, 3, A,
                                    &Q, &R, nullptr, cc ) >= 0);

    // print out Q matrix
    cholmod_l_write_sparse( stdout, Q , NULL, NULL, cc );

    // free everything and finish CHOLMOD
    klu_l_free_symbolic (&KS, &kcommon);
    klu_l_free_numeric  (&KN, &kcommon);

    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_dense  (&Adense, cc) ;
    cholmod_l_free_sparse (&G, cc) ;
    cholmod_l_free_sparse (&Q, cc) ;
    cholmod_l_free_sparse (&R, cc) ;
    cholmod_l_free_triplet (&Gct, cc) ;
    cholmod_l_finish (cc) ;

    return (0) ;
}
