// Describing expectations for a valid sparse matrix library

#ifndef SPARSELIB_CONCEPT_HPP
#define SPARSELIB_CONCEPT_HPP

#include <type_traits>
#include <iostream>

// First, some preliminaries (no predefined Concepts are supplied in library yet)
template<typename T> concept bool ForwardIterator =
    requires( T a, T b ) {
    { a != b }
    { a == b }
    { ++a }
    { *a }
};

template<typename T, typename U> concept bool DereferencesTo =
    requires( T a ) {
    { *a } -> U;
};

template<typename T, typename ...Args> concept bool ConstructibleFrom =
    std::is_constructible<T, Args...>::value ||     // constructors
    requires(Args... args) { { T{args...} } -> T }; // aggregate initialization

template<typename T> concept bool ForwardIterator =
    std::is_convertible<typename std::iterator_traits<T>::iterator_category,
                        std::forward_iterator_tag>::value;

template<typename T, typename U> concept bool Same =
    std::is_same<T, U>::value;

template<typename T, typename U> concept bool IteratorRange =
    Same<T, U> && ForwardIterator<T>;

template<typename L> concept bool SparseLibrary =
    requires( typename L::sparsemat_t mat,
              typename L::lu_t lu,
              typename L::qr_t qr,
              std::ostream& os
        ) {
    // inner types we expect
    typename L::sparsemat_t;
    typename L::index_t;
    typename L::value_t;
    typename L::triplet_t;
    typename L::lu_t;
    typename L::qr_t;
    
    // triplets can be constructed
    requires ConstructibleFrom<typename L::triplet_t, typename L::index_t, typename L::index_t, typename L::value_t>;

    // we can construct QR and LU objects from a sparse matrix
    requires ConstructibleFrom<typename L::lu_t, typename L::sparsemat_t>;
    requires ConstructibleFrom<typename L::qr_t, typename L::sparsemat_t>;

    // The LU decomposition can perform a solve against a sparse matrix with a sparse result
    { lu.solve( mat ) } -> typename L::sparsemat_t ;

    // The QR decomposition can return a Q of a type convertible to sparse
    { qr.Q() } -> typename L::sparsemat_t;

    // The Q returned from the QR can multiply a sparse matrix on the left, producing another
    // note this does *not* require that the result be a sparse matrix - just that you can convert it to one
    { qr.Q() * mat } -> typename L::sparsemat_t;

    // you can stream out a sparse matrix
    { os << mat } -> std::ostream &;

};

#endif // SPARSELIB_CONCEPT_HPP
