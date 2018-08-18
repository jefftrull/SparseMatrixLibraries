cmake_minimum_required( VERSION 3.5 )

PROJECT( SparseMatrixLibraries )

set( CMAKE_CXX_STANDARD 14 )     # it's time, people

# Create a compilation database (compile_commands.json) for the use of Clang tools
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )

# You can use SuiteSparse/CSparse out of regular Ubuntu system paths if you install
# libsuitesparse-dev.
# If you want to use a recent version of SuiteSparse, download and build it, then
# point CMake at it from the command line with -DSUITESPARSE_ROOT=/path/to/dir

# Finding SuiteSparse manually because there seems to be no standard CMake module

# try user hint first
if( SUITESPARSE_ROOT )
  # Assuming CSparse and SuiteSparse come from the same SuiteSparse distribution
  find_path( SUITESPARSE_INCLUDE SuiteSparseQR.hpp
             PATHS ${SUITESPARSE_ROOT}/include  )
  # also point linker to these directories
  link_directories(
    ${SUITESPARSE_ROOT}/CXSparse/Lib
    ${SUITESPARSE_ROOT}/CHOLMOD/Lib
    ${SUITESPARSE_ROOT}/CCOLAMD/Lib
    ${SUITESPARSE_ROOT}/COLAMD/Lib
    ${SUITESPARSE_ROOT}/AMD/Lib
    ${SUITESPARSE_ROOT}/SPQR/Lib
    ${SUITESPARSE_ROOT}/KLU/Lib
    ${SUITESPARSE_ROOT}/BTF/Lib
    ${SUITESPARSE_ROOT}/CAMD/Lib
    ${SUITESPARSE_ROOT}/SuiteSparse_config
  )

  # Metis is a little funny
  file( GLOB METIS_BUILD_DIRS ${SUITESPARSE_ROOT}/metis-5.1.0/build/*/libmetis )
  find_library( METIS_LIB metis PATHS ${METIS_BUILD_DIRS} )

else()
  # try the system default paths as used in Ubuntu
  find_path( SUITESPARSE_INCLUDE suitesparse/SuiteSparseQR.hpp )
  if( SUITESPARSE_INCLUDE )
    set( SUITESPARSE_INCLUDE "${SUITESPARSE_INCLUDE}/suitesparse" )
  endif()
endif()

if( SUITESPARSE_INCLUDE )
  include_directories( SYSTEM ${SUITESPARSE_INCLUDE} )
else()
  message( FATAL_ERROR "could not find SuiteSparse headers" )
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror" )

set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer" )

# CSparse and SPQR examples
add_executable( cs_prima csprima.cpp )
add_executable( ss_prima ssprima.cpp )

# special properties for CSparse
target_link_libraries( cs_prima cxsparse )

# special properties for SPQR
target_link_libraries( ss_prima klu btf spqr cholmod ccolamd colamd amd )
target_link_libraries( ss_prima openblas )  # other blas should be OK here too
target_link_libraries( rqrss klu btf spqr cholmod ccolamd colamd amd openblas suitesparseconfig ${METIS_LIB} camd)
if( SUITESPARSE_ROOT )
  target_link_libraries( ss_prima suitesparseconfig )
endif()
if( METIS_LIB )
  target_link_libraries( ss_prima ${METIS_LIB} camd )
endif()

# finally Eigen
if( EIGEN3_INCLUDE_DIR )
  set( CMAKE_MODULE_PATH ${EIGEN3_INCLUDE_DIR}/cmake )
  list( REMOVE_DUPLICATES CMAKE_MODULE_PATH )

  find_package( Eigen3 )
  if( NOT EIGEN3_FOUND )
    message( FATAL_ERROR "EIGEN3_INCLUDE_DIR supplied but Eigen was not found" )
  endif()
  # TODO: utilize other find modules (cholmod, blas, Metis, etc.) supplied with Eigen

  include_directories( SYSTEM ${EIGEN3_INCLUDE_DIR} )
endif()

add_executable( ei_prima eisprima.cpp )

# experiment: integrate our code using a single main() and a Policy wrapping each library
add_subdirectory( policies )
