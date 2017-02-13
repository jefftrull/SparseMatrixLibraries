# Experiments with Policy classes
# I create a single interface, described as a Concept, and implement it three times, one for each "library"
# then the main code simply selects (at compile time) one of the policies and runs with it

# In order to use the Concepts TS we need a recent version of gcc
if( ( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" ) AND NOT ( CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.2.0 ) )
  # Only difference between Eigen/CSparse/SuiteSparse builds will be a preprocessor definition
  add_executable( epolicy policy_experiment.cpp )
  target_compile_options( epolicy PUBLIC -fconcepts )
  target_compile_definitions( epolicy PUBLIC USE_EIGEN )
endif()

# TODO: Boost.Concepts alternative
