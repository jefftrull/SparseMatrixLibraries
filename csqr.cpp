// Using CSparse to compute a QR decomposition
// based on example from CSparse docs

#include <vector>
#include <tuple>
#include <cassert>
#include <iostream>

#include <cs.h>

struct triplet {
    int row;
    int col;
    double value;
};

int main ()
{
    using namespace std;

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

    // create a triplet matrix
    cs * T = cs_spalloc(15, 3, entries.size(), 1, 1);

    for ( size_t i = 0; i < entries.size(); ++i) {
        assert(cs_entry(T, entries[i].row, entries[i].col, entries[i].value));
    }

    // create a "cs" structure from the triplet matrix
    cs *A = cs_compress(T); cs_spfree(T);
                      
    // run QR
    // symbolic first
    css * S = cs_sqr( 3,   // order amd(A'*A)
                      A,
                      1 ); // indicates is for the use of a QR decomposition (as opposed to LU)

    csn * result = cs_qr(A, S);

    using csi = int;

    // must reverse the permutation used for solves to generate Q
    csi * P = cs_pinv(S->pinv, 15);

    double * x = (double *)cs_calloc(S->m2, sizeof(double));

    cs* V = result->L;

    // generate Q:

    // create a *dense* identity matrix of the right size
    double * I = (double *)cs_calloc (15*3, sizeof(double));
    I[0*15+0] = 1;
    I[1*15+1] = 1;
    I[2*15+2] = 1;

    // apply householder vectors to I to produce Q
    double * Q = I;

    // following the pattern from cs_qrsol.c:

    for ( csi j = 0; j <= 2; j++) {
        double * col = Q + 15*j;
        // make x a copy of the jth column of I
        std::copy(col, col+15, x);

        // apply the Householder vectors that comprise Q
        for (csi k = j; k >= 0; k--) {
            cs_happly( V, k, result->B[k], x );
        }
        // apply the row permutation
        cs_ipvec( P, x, col, 15 );

    }

    std::cout << "Q=[\n";
    for ( csi i = 0; i < 15; ++i) {
        std::cout << Q[0*15+i] << ", " << Q[1*15+i] << ", " << Q[2*15+i] << ";\n";
    }
    std::cout << "]\n";
            
    // clean up so valgrind etc. will be happy
    cs_free(P);
    cs_free(x);
    cs_free(I);
    cs_nfree(result);
    cs_sfree(S);
    cs_spfree(A);
    


    return (0) ;
}
