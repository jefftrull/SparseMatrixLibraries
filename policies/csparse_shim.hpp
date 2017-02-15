#include <memory>
#include <cassert>
#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>

#include <boost/iterator/iterator_facade.hpp>

#include <cs.h>

struct CSparseShim {
    using index_t = CS_INT;     // for options, refer to CS_LONG and CS_COMPLEX in cs.h
    using value_t = CS_ENTRY;

    struct triplet_t {
        index_t row;
        index_t col;
        value_t value;
    };

    // RAII memory management for C data
    struct cs_deleter {
        void operator()(cs *p) {
            cs_spfree(p);
        }
        void operator()(csn *p) {
            cs_nfree(p);
        }
        void operator()(css *p) {
            cs_sfree(p);
        }
        template<typename T>
        void operator()(T *p) {
            cs_free(p);
        }
    };

    template<typename T> using cs_unique_ptr = std::unique_ptr<T, cs_deleter>;
    template<typename T> using cs_shared_ptr = std::shared_ptr<T>;
    template<typename T>
    static cs_shared_ptr<T>
    make_cs_shared_ptr(T* p) { return cs_shared_ptr<T>(p, cs_deleter()); }


    // create a forward iterator for nonzeros within a CSparse matrix
    struct sparse_entry_iterator
        : boost::iterator_facade<sparse_entry_iterator,
                                 triplet_t const,
                                 boost::forward_traversal_tag> {
        sparse_entry_iterator() {}
        sparse_entry_iterator( cs_shared_ptr<cs> mat )
            : mat_(std::move(mat)), entry_{index_t(0), index_t(0)} {
            advance_to_valid();
        }

    private:

        // iterator_facade requirements

        friend class boost::iterator_core_access;

        void increment();

        bool equal( sparse_entry_iterator const& other ) const;

        triplet_t const & dereference() const {
            return entry_;
        }

        // find the next nonzero entry and set the dereference value
        void advance_to_valid();

        cs_shared_ptr<cs> mat_;
        triplet_t entry_;     // for supplying references
    };


    struct sparsemat_t;

    static sparsemat_t
    dense_to_sparse( std::vector<value_t> const& d, index_t rows, index_t cols );

    struct sparsemat_t {
        template<typename Iter>
        sparsemat_t(index_t rows, index_t cols, Iter start, Iter end) {
            // create a triplet matrix
            auto TG = cs_unique_ptr<cs>(cs_spalloc(rows, cols, std::distance(start, end), 1, 1));
            for ( auto it = start; it < end; ++it ) {
                assert(cs_entry(TG.get(), it->row, it->col, it->value));
            }
                
            // create a "cs" structure from the triplet matrix
            mat_ = make_cs_shared_ptr(cs_compress(TG.get()));
        }

        sparsemat_t( cs_shared_ptr<cs> mat_cs ) : mat_(std::move(mat_cs)) {}

        index_t rows() const { return wrapped()->m; }
        index_t cols() const { return wrapped()->n; }

        sparse_entry_iterator nonzero_begin() const {
            return sparse_entry_iterator(mat_);
        }
        sparse_entry_iterator nonzero_end() const {
            return sparse_entry_iterator();
        }

        friend sparsemat_t operator*(sparsemat_t const& a, sparsemat_t const& b);
        friend std::ostream& operator<<(std::ostream& os, sparsemat_t const & m);

        cs_shared_ptr<const cs> wrapped() const {
            return mat_;
        }

    private:
        cs_shared_ptr<cs> mat_;      // we have to share this structure with LU and QR objects
    };

    struct lu_t {
        lu_t( sparsemat_t const & mat )
            : symbolic_( cs_sqr( 3, mat.wrapped().get(), 0 )),
              numeric_ ( cs_lu ( mat.wrapped().get(), symbolic_.get(),
                                 std::numeric_limits<value_t>::epsilon() )) {}

        sparsemat_t solve(sparsemat_t const& rhs) const;

    private:
        cs_unique_ptr<css> symbolic_;
        cs_unique_ptr<csn> numeric_;

    };

    struct qr_t {
        qr_t( sparsemat_t const & mat )
            : symbolic_( cs_sqr( 3, mat.wrapped().get(), 1 ) ),
              numeric_ ( cs_qr ( mat.wrapped().get(), symbolic_.get() ) ),
              rows_    ( mat.wrapped()->m ),
              cols_    ( mat.wrapped()->n ) {}

        sparsemat_t Q() const;

    private:

        index_t rows_, cols_;

        cs_unique_ptr<css> symbolic_;
        cs_unique_ptr<csn> numeric_;
    };

};
