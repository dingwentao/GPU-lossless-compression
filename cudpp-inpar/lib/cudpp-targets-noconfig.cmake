#----------------------------------------------------------------
# Generated CMake target import file for configuration "".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cudpp" for configuration ""
set_property(TARGET cudpp APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cudpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "/usr/local/cuda/lib64/libcudart.so"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcudpp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cudpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_cudpp "${_IMPORT_PREFIX}/lib/libcudpp.a" )

# Import target "cudpp_hash" for configuration ""
set_property(TARGET cudpp_hash APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(cudpp_hash PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LINK_INTERFACE_LIBRARIES_NOCONFIG "/usr/local/cuda/lib64/libcudart.so;cudpp"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libcudpp_hash.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS cudpp_hash )
list(APPEND _IMPORT_CHECK_FILES_FOR_cudpp_hash "${_IMPORT_PREFIX}/lib/libcudpp_hash.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
