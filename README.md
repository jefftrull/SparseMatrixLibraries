# Usage Examples for Sparse Matrix Libraries

As an outgrowth of some debugging I've been doing I made test cases of the same problem with several different libraries:

* [Eigen](http://eigen.tuxfamily.org/dox/group__TutorialSparse.html)
* [CSparse](https://people.sc.fsu.edu/~jburkardt/c_src/csparse/csparse.html)
* [SuiteSparse](http://faculty.cse.tamu.edu/davis/suitesparse.html) with a tiny bit of [KLU](https://pdfs.semanticscholar.org/4fd6/3bec7022dc25d2f8ca2ad6093c7774ef4283.pdf)

It seemed like a good idea to upload these test cases for public reference.

In addition I have used those three different implementations as an example of a Policy pattern,
with the supplied implementations checked against Concepts using either the [Concepts TS](http://en.cppreference.com/w/cpp/language/constraints) or the [Boost Concept Check Library](http://www.boost.org/doc/libs/1_63_0/libs/concept_check/concept_check.htm), depending on compiler support. For details see the code in the `policies/` directory.
