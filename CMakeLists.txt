cmake_minimum_required( VERSION 3.5 )

PROJECT( SparseMatrixLibraries )

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

# CSparse and SPQR examples
add_executable( cs_qr csqr.cpp )
add_executable( sp_qr spqr.cpp )

# shared properties
foreach( x cs_qr sp_qr )
  set_target_properties( ${x} PROPERTIES
    CXX_STANDARD 14
  )
endforeach()

# special properties for CSparse
target_link_libraries( cs_qr cxsparse )

# special properties for SPQR
target_link_libraries( sp_qr spqr cholmod ccolamd colamd amd )
target_link_libraries( sp_qr openblas )  # other blas should be OK here too
if( SUITESPARSE_ROOT )
  target_link_libraries( sp_qr suitesparseconfig )
endif()
if( METIS_LIB )
  target_link_libraries( sp_qr ${METIS_LIB} camd )
endif()
