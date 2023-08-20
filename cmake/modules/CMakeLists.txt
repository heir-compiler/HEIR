# Generate a list of CMake library targets so that other CMake projects can
# link against them.
set(HEIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/heir)
set(heir_cmake_builddir "${CMAKE_BINARY_DIR}/${HEIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with mlir/cmake/CMakeLists.txt!
set(MLIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir)
set(mlir_cmake_builddir "${MLIR_BINARY_DIR}/${MLIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
get_property(HEIR_EXPORTS GLOBAL PROPERTY HEIR_EXPORTS)
export(TARGETS ${HEIR_EXPORTS} FILE ${heir_cmake_builddir}/HEIRTargets.cmake)

get_property(HEIR_ALL_LIBS GLOBAL PROPERTY HEIR_ALL_LIBS)
get_property(HEIR_DIALECT_LIBS GLOBAL PROPERTY HEIR_DIALECT_LIBS)
get_property(HEIR_CONVERSION_LIBS GLOBAL PROPERTY HEIR_CONVERSION_LIBS)
get_property(HEIR_TRANSLATION_LIBS GLOBAL PROPERTY HEIR_TRANSLATION_LIBS)
get_property(HEIR_ANALYSIS_LIBS GLOBAL PROPERTY HEIR_ANALYSIS_LIBS)

# Generate HEIRConfig.cmake for the build tree.
set(HEIR_CONFIG_CMAKE_DIR "${heir_cmake_builddir}")
set(HEIR_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(HEIR_CONFIG_MLIR_CMAKE_DIR "${mlir_cmake_builddir}")
set(HEIR_CONFIG_LIBRARY_DIRS "${HEIR_LIBRARY_DIR}")
set(HEIR_CONFIG_BINARY_DIR "${HEIR_BINARY_DIR}")
set(HEIR_CONFIG_TOOLS_DIR "${HEIR_TOOLS_DIR}")
set(HEIR_CONFIG_INCLUDE_EXPORTS "include(\"\${HEIR_CMAKE_DIR}/HEIRTargets.cmake\")")
set(HEIR_CONFIG_INCLUDE_DIRS
  "${HEIR_SOURCE_DIR}/include"
  "${HEIR_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/HEIRConfig.cmake.in
  ${heir_cmake_builddir}/HEIRConfig.cmake
  @ONLY)
set(HEIR_CONFIG_CMAKE_DIR)
set(HEIR_CONFIG_LLVM_CMAKE_DIR)
set(HEIR_CONFIG_MLIR_CMAKE_DIR)
set(HEIR_CONFIG_LIBRARY_DIRS)
set(HEIR_CONFIG_BINARY_DIR)
set(HEIR_CONFIG_TOOLS_DIR)
set(HEIR_CONFIG_INCLUDE_EXPORTS)
set(HEIR_CONFIG_INCLUDE_DIRS)

# Generate HEIRConfig.cmake for the install tree.
set(HEIR_CONFIG_CODE "
# Compute the installation prefix from this HEIRConfig.cmake file location.
get_filename_component(HEIR_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${HEIR_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(HEIR_CONFIG_CODE "${HEIR_CONFIG_CODE}
get_filename_component(HEIR_INSTALL_PREFIX \"\${HEIR_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(HEIR_CONFIG_CMAKE_DIR "\${HEIR_INSTALL_PREFIX}/${HEIR_INSTALL_PACKAGE_DIR}")
set(HEIR_CONFIG_LLVM_CMAKE_DIR "\${HEIR_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(HEIR_CONFIG_MLIR_CMAKE_DIR "\${HEIR_INSTALL_PREFIX}/${MLIR_INSTALL_PACKAGE_DIR}")
set(HEIR_CONFIG_LIBRARY_DIRS "\${HEIR_INSTALL_PREFIX}/lib")
set(HEIR_CONFIG_BINARY_DIR "\${HEIR_INSTALL_PREFIX}")
set(HEIR_CONFIG_TOOLS_DIR "\${HEIR_INSTALL_PREFIX}/bin")
set(HEIR_CONFIG_INCLUDE_EXPORTS "include(\"\${HEIR_CMAKE_DIR}/HEIRTargets.cmake\")")
set(HEIR_CONFIG_INCLUDE_DIRS
  "\${HEIR_INSTALL_PREFIX}/include"
  )
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/HEIRConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/HEIRConfig.cmake
    @ONLY)
set(HEIR_CONFIG_CODE)
set(HEIR_CONFIG_CMAKE_DIR)
set(HEIR_CONFIG_LLVM_CMAKE_DIR)
set(HEIR_CONFIG_MLIR_CMAKE_DIR)
set(HEIR_CONFIG_LIBRARY_DIRS)
set(HEIR_CONFIG_BINARY_DIR)
set(HEIR_CONFIG_TOOLS_DIR)
set(HEIR_CONFIG_INCLUDE_EXPORTS)
set(HEIR_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the HEIR parts as well
  # Include the cmake files so other tools can use heir-tblgen, etc.
  install(EXPORT HEIRTargets DESTINATION ${HEIR_INSTALL_PACKAGE_DIR}
          COMPONENT heir-cmake-exports)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/HEIRConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddHEIR.cmake
    DESTINATION ${HEIR_INSTALL_PACKAGE_DIR}
    COMPONENT heir-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(heir-cmake-exports)
    add_llvm_install_targets(install-heir-cmake-exports
                             COMPONENT heir-cmake-exports)
  endif()
endif()