# Experiments with Policy classes
# I create a single interface, described as a Concept, and implement it three times, one for each "library"
# then the main code simply selects (at compile time) one of the policies and runs with it

find_package( Boost 1.63 )

# In order to use the Concepts TS we need a recent version of gcc
if( ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" ) AND NOT ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.2.0 ) )
  # Only difference between Eigen/CSparse/SuiteSparse builds will be a preprocessor definition
  add_executable( epolicy policy_experiment.cpp )
  target_compile_options( epolicy PUBLIC -fconcepts )
  target_compile_definitions( epolicy PUBLIC USE_EIGEN )

  if ( SUITESPARSE_ROOT )
    add_executable( cpolicy policy_experiment.cpp csparse_shim.cpp )
    target_compile_options( cpolicy PUBLIC -fconcepts )
    target_compile_definitions( cpolicy PUBLIC USE_CSPARSE )
    target_include_directories( cpolicy PUBLIC ${Boost_INCLUDE_DIRS} )
    target_link_libraries( cpolicy cxsparse )
  endif()
endif()

# TODO: Boost.Concepts alternative
