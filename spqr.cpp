// Illustration of (direct) use of SuiteSparse and SQPR
// based on example from SuiteSparse docs

#include <vector>
#include <tuple>
#include <cassert>

#include <SuiteSparseQR.hpp>

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

    vector<triplet> entries{
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

    // now load into a Cholmod "triplet matrix"
    cholmod_triplet * ct = cholmod_l_allocate_triplet(
        15, 3, entries.size(),
        0,  // stype: both upper and lower are stored
        CHOLMOD_REAL,
        cc);
    for ( size_t i = 0; i < entries.size(); ++i) {
        reinterpret_cast<long *>(ct->i)[i] = entries[i].row;
        reinterpret_cast<long *>(ct->j)[i] = entries[i].col;
        reinterpret_cast<double *>(ct->x)[i] = entries[i].value;
    }
    ct->nnz = entries.size();

    // convert to cholmod_sparse
    cholmod_sparse * A = cholmod_l_triplet_to_sparse(ct, entries.size(), cc);
    cholmod_l_write_sparse( stdout, A , NULL, NULL, cc );

    // run QR
    cholmod_sparse * Q;   // outputs
    cholmod_sparse * R;
    SuiteSparse_long * E;
    assert( SuiteSparseQR<double> ( SPQR_ORDERING_DEFAULT, SPQR_DEFAULT_TOL, 3, A,
                                    &Q, &R, &E, cc ) >= 0);

    // print out Q matrix
    cholmod_l_write_sparse( stdout, Q , NULL, NULL, cc );

    // free everything and finish CHOLMOD
    cholmod_l_free_sparse (&A, cc) ;
    cholmod_l_free_sparse (&Q, cc) ;
    cholmod_l_free_sparse (&R, cc) ;
    // cholmod_l_free_long (&E, cc) ;
    cholmod_l_free_triplet (&ct, cc) ;
    cholmod_l_finish (cc) ;

    return (0) ;
}
