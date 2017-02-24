// An implementation of the SparseLib concept using the Boost Concept Check library
// instead of the Concepts TS

#include <boost/concept_check.hpp>

// a Concept Checking Class for SparseLib

template<class L>
struct SparseLibrary {
    // expected types
    using sparsemat_t = typename L::sparsemat_t;
    using index_t     = typename L::index_t;
    using value_t     = typename L::value_t;
    using triplet_t   = typename L::triplet_t;
    using lu_t        = typename L::lu_t;
    using qr_t        = typename L::qr_t;

    // Expressions that need to be valid
    BOOST_CONCEPT_USAGE(SparseLibrary) {
        // triplets can be constructed
        triplet_t t{idx_, idx_, value_};

        // QR and LU objects can be constructed from sparse matrices
        qr_t qr(mat_);
        lu_t lu(mat_);

        // The LU decomposition can perform a solve against a sparse matrix with a sparse result
        sparsemat_t s1 = lu_.solve( mat_ ) ;

        // The QR decomposition can return a Q of a type convertible to sparse
        sparsemat_t s2 = qr_.Q() ;

        // A Q object can multiply a sparse matrix on the right, resulting in a sparse matrix
        sparsemat_t s3 = qr_.Q() * mat_ ;

        // Sparse matrices can be streamed out
        std::ostream& os = (os_ << mat_);

        // If, like me, you turn on -Wunused-variable and -Werror you will need:
        (void)t;  (void)os;

    }
private:
    triplet_t     t_;
    sparsemat_t   mat_;
    lu_t          lu_;
    qr_t          qr_;
    index_t       idx_;
    value_t       value_;
    std::ostream  os_;

};
