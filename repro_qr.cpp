// Seeing what Suitesparse does with the same input as our problem Eigen sparse QR solve

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

    vector<triplet> At{
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

    vector<triplet> bt{
        {0, 0, 10.3612221487294},
        {1, 0, -2.27836032745297},
        {2, 0, -10.3179355124181},
        {3, 0, -7.49344052343591}};
        
    // now load into Cholmod "triplet matrix"
    auto Act = make_ss_ptr(
        cholmod_l_allocate_triplet(
            4, 5, At.size(),
            0, CHOLMOD_REAL, cc),
        ccommon);
    for ( size_t i = 0; i < At.size(); ++i) {
        reinterpret_cast<long *>(Act->i)[i] = At[i].row;
        reinterpret_cast<long *>(Act->j)[i] = At[i].col;
        reinterpret_cast<double *>(Act->x)[i] = At[i].value;
    }
    Act->nnz = At.size();

    // turn triplet matrix into sparse
    auto A = make_ss_ptr(
        cholmod_l_triplet_to_sparse(Act.get(), At.size(), cc),
        ccommon);

    // b must be dense
    auto b = make_ss_ptr(
        cholmod_l_zeros( 4, 1, CHOLMOD_REAL, cc ),
        ccommon);
    for ( size_t i = 0; i < bt.size(); ++i) {
        reinterpret_cast<double *>(b->x)[4*bt[i].col+bt[i].row] = bt[i].value;
    }

    auto x = make_ss_ptr(
        SuiteSparseQR<double>(SPQR_ORDERING_COLAMD, SPQR_DEFAULT_TOL, A.get(), b.get(), cc),
        ccommon);

    if (cc->status != CHOLMOD_OK) {
        printf("failed!!!\n");
    }

    // print out solution
    printf("using A=\n");
    cholmod_l_write_sparse( stdout, A.get(), nullptr, nullptr, cc );
    printf("and b=\n");
    cholmod_l_write_dense( stdout, b.get(), nullptr, cc );
    printf("we got x=\n");
    cholmod_l_write_dense( stdout, x.get(), nullptr, cc );

    // now also find what permutation SS used:
    cholmod_sparse * R;
    SuiteSparse_long * E;
    auto rank = SuiteSparseQR<double>(SPQR_ORDERING_COLAMD, SPQR_DEFAULT_TOL, 4,
                                      A.get(), &R, &E, cc);

    printf("rank is %ld ", rank);
    if (E) {
        printf("and permutation is:\n");
        for (int i = 0; i < 5; ++i) {
            printf("%ld ", E[i]);
        }
    }
    printf("\n");
        
    // Now find out what the permutation is if SuiteSparse's internal colamd is used on its own
    // (code *actually* does a bunch of other stuff first, including moving dense or null
    // columns and rows to the end, which in our test case nearly eliminates the work for colamd)
    // use the approach from Eigen
    auto Alen = colamd_recommended(11, 4, 5);
    std::vector<long> p((long*)A->p, (long*)A->p + 6);
    std::vector<long> A_idx(Alen);
    std::copy((long*)A->i, (long*)A->i + 11, A_idx.data());
    double knobs[COLAMD_KNOBS];
    long   stats[COLAMD_STATS];
    colamd_l(4, 5, Alen, A_idx.data(), p.data(), knobs, stats);

    // turn permutation into something we understand
    std::vector<long> perm(5);
    for (long i = 0; i < 5; ++i)
    {
        perm[p[i]] = i;
    }

    printf("permutation from plain colamd run is:\n");
    for (int i = 0; i < 5; ++i) {
      printf("%ld ", perm[i]);
    }
    printf("\n");
}
