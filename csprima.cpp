// Using CSparse to compute a QR decomposition
// based on example from CSparse docs

#include <vector>
#include <tuple>
#include <cassert>
#include <iostream>
#include <algorithm>

#include <cs.h>

struct triplet {
    int row;
    int col;
    double value;
};

int main ()
{
    using namespace std;

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

    // create a triplet matrix
    cs * TG = cs_spalloc(15, 15, Gentries.size(), 1, 1);

    for ( size_t i = 0; i < Gentries.size(); ++i) {
        assert(cs_entry(TG, Gentries[i].row, Gentries[i].col,Gentries[i].value));
    }

    // create a "cs" structure from the triplet matrix
    cs *G = cs_compress(TG); cs_spfree(TG);

    // We need to solve G\B but CSparse only has canned routines for B dense, so
    // we will start there and then convert.  Inefficient, though :(

    // initialize A (the eventual result) with the contents of B
    vector<double> Adense(15*3);
    fill(Adense.begin(), Adense.end(), 0);
    for ( size_t i = 0; i < Bentries.size(); ++i) {
        Adense[15*Bentries[i].col+Bentries[i].row] = Bentries[i].value;
    }

    // calculate LU factorization of G
    vector<double> workspace(15);
    // symbolic factorization first
    css * S = cs_sqr ( 3,   // order amd(A'*A)
                       G,
                       0 ); // for use by LU,  not QR
    // then numeric
    csn * N = cs_lu ( G, S, numeric_limits<double>::epsilon() );

    // now use the result to solve G^-1*B one column at a time
    for ( int j = 0; j < 3; j++ ) {
        cs_ipvec ( N->pinv, Adense.data()+15*j, workspace.data(), 15 );
        cs_lsolve ( N->L, workspace.data() ) ;
        cs_usolve ( N->U, workspace.data() ) ;
        cs_ipvec ( S->q, workspace.data(), Adense.data()+15*j, 15 );
    }

    // produce sparse A from dense entries
    size_t nonzeroA_count =
        count_if(Adense.begin(), Adense.end(),
                      [](double x) { return (x != 0.0); });

    cs * TA = cs_spalloc(15, 3, nonzeroA_count, 1, 1);

    for ( size_t j = 0; j < 3; ++j) {
        for ( size_t i = 0; i < 15; ++i) {
            if ( Adense[15*j+i] != 0 ) {
                assert(cs_entry(TA, i, j, Adense[15*j+i]));
            }
        }
    }
    cs *A = cs_compress(TA); cs_spfree(TA);

    // run QR on A
    cs_sfree(S);
    S = cs_sqr( 3,
                A,
                1 ); // for QR decomposition

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
    cs_nfree(N);
    cs_spfree(A);
    cs_spfree(G);
    


    return (0) ;
}
