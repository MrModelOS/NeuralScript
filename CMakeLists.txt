cmake_minimum_required(VERSION 3.20)
project(NeuralScript VERSION 1.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

option(NS_BUILD_TESTS "Build the test suite" ON)
option(NS_ENABLE_CUDA "Enable CUDA backend integration tests (requires a CUDA toolkit and, to run, a device)" OFF)

add_library(ns_core STATIC
    src/lexer/lexer.cpp
    src/lexer/token.cpp
    src/parser/parser.cpp
    src/parser/ast.cpp
    src/typechecker/shape_checker.cpp
    src/typechecker/type_system.cpp
    src/mlir/mlir_dialect.cpp
    src/mlir/mlir_compiler.cpp
    src/mlir/fusion.cpp
    src/mlir/eval.cpp
    src/codegen/codegen.cpp
    src/codegen/cuda_backend.cpp
    src/optim/muon.cpp
    src/training/trainer.cpp
)

target_include_directories(ns_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)

add_executable(nsc src/main.cpp)
target_link_libraries(nsc ns_core)

# ---- Install / export ----------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ns_core nsc
        EXPORT nsCoreTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

install(DIRECTORY include/ns
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT nsCoreTargets
        FILE nsCoreTargets.cmake
        NAMESPACE NeuralScript::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralScript)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/NeuralScriptConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralScriptConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralScript)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralScriptConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralScriptConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/NeuralScriptConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/NeuralScript)

if(NS_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()