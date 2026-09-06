cmake_minimum_required(VERSION 3.20)
project(NeuralScript VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

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
    src/autodiff/autodiff.cpp
    src/optim/muon.cpp
    src/training/trainer.cpp
)

target_include_directories(ns_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(nsc src/main.cpp)
target_link_libraries(nsc ns_core)

enable_testing()
add_subdirectory(tests)
