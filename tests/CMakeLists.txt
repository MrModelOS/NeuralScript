enable_testing()

# Asserts must be active even in Release builds so tests fail loudly.
add_compile_options(-UNDEBUG)

add_executable(test_lexer test_lexer.cpp)
target_link_libraries(test_lexer ns_core)
add_test(NAME test_lexer COMMAND test_lexer)

add_executable(test_parser test_parser.cpp)
target_link_libraries(test_parser ns_core)
add_test(NAME test_parser COMMAND test_parser)

add_executable(test_typechecker test_typechecker.cpp)
target_link_libraries(test_typechecker ns_core)
add_test(NAME test_typechecker COMMAND test_typechecker)

add_executable(test_codegen test_codegen.cpp)
target_link_libraries(test_codegen ns_core)
add_test(NAME test_codegen COMMAND test_codegen)

add_executable(test_codegen_cpu test_codegen_cpu.cpp)
target_link_libraries(test_codegen_cpu ns_core)
add_test(NAME test_codegen_cpu COMMAND test_codegen_cpu)
set_tests_properties(test_codegen_cpu PROPERTIES TIMEOUT 120)

add_executable(test_transformer test_transformer.cpp)
target_link_libraries(test_transformer ns_core)
add_test(NAME test_transformer COMMAND test_transformer)
set_tests_properties(test_transformer PROPERTIES TIMEOUT 120)

add_executable(test_datamove test_datamove.cpp)
target_link_libraries(test_datamove ns_core)
add_test(NAME test_datamove COMMAND test_datamove)
set_tests_properties(test_datamove PROPERTIES TIMEOUT 120)

add_executable(test_dsl_datamove test_dsl_datamove.cpp)
target_link_libraries(test_dsl_datamove ns_core)
add_test(NAME test_dsl_datamove COMMAND test_dsl_datamove)
set_tests_properties(test_dsl_datamove PROPERTIES TIMEOUT 120)
set_tests_properties(test_transformer test_datamove test_dsl_datamove PROPERTIES SKIP_RETURN_CODE 77)

add_executable(test_moe test_moe.cpp)
target_link_libraries(test_moe ns_core)
add_test(NAME test_moe COMMAND test_moe)
set_tests_properties(test_moe PROPERTIES TIMEOUT 120 SKIP_RETURN_CODE 77)

add_executable(test_dynamic_dims test_dynamic_dims.cpp)
target_link_libraries(test_dynamic_dims ns_core)
add_test(NAME test_dynamic_dims COMMAND test_dynamic_dims)

add_executable(test_fusion test_fusion.cpp)
target_link_libraries(test_fusion ns_core)
add_test(NAME test_fusion COMMAND test_fusion)

add_executable(test_optim test_optim.cpp)
target_link_libraries(test_optim ns_core)
add_test(NAME test_optim COMMAND test_optim)

add_executable(test_runtime test_runtime.cpp)
target_link_libraries(test_runtime ns_core)
target_compile_definitions(test_runtime PRIVATE
    NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(NAME test_runtime COMMAND test_runtime)
set_tests_properties(test_runtime PROPERTIES TIMEOUT 120)

add_executable(test_train test_train.cpp)
target_link_libraries(test_train ns_core)
add_test(NAME test_train COMMAND test_train)
set_tests_properties(test_train PROPERTIES TIMEOUT 120)

add_executable(test_aot_train test_aot_train.cpp)
target_link_libraries(test_aot_train ns_core)
target_compile_definitions(test_aot_train PRIVATE
    NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(NAME test_aot_train COMMAND test_aot_train)
set_tests_properties(test_aot_train PROPERTIES TIMEOUT 120)

add_executable(test_layernorm_train test_layernorm_train.cpp)
target_link_libraries(test_layernorm_train ns_core)
target_compile_definitions(test_layernorm_train PRIVATE
    NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(NAME test_layernorm_train COMMAND test_layernorm_train)
set_tests_properties(test_layernorm_train PROPERTIES TIMEOUT 120)

add_executable(test_embedding_train test_embedding_train.cpp)
target_link_libraries(test_embedding_train ns_core)
target_compile_definitions(test_embedding_train PRIVATE
    NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(NAME test_embedding_train COMMAND test_embedding_train)
set_tests_properties(test_embedding_train PROPERTIES TIMEOUT 120)

add_executable(test_moe_train test_moe_train.cpp)
target_link_libraries(test_moe_train ns_core)
target_compile_definitions(test_moe_train PRIVATE
    NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(NAME test_moe_train COMMAND test_moe_train)
set_tests_properties(test_moe_train PROPERTIES TIMEOUT 120)

add_executable(test_attention_train test_attention_train.cpp)
target_link_libraries(test_attention_train ns_core)
target_compile_definitions(test_attention_train PRIVATE
    NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include")
add_test(NAME test_attention_train COMMAND test_attention_train)
set_tests_properties(test_attention_train PROPERTIES TIMEOUT 120)

# CUDA backend integration is opt-in: adding it unconditionally would make
# ctest FAIL/hang on machines without the CUDA toolkit or a device. With
# -DNS_ENABLE_CUDA=ON the test compiles the emitted .cu with nvcc (if present)
# and runs on a device if one exists; otherwise it self-skips via exit code
# 77 (see SKIP_RETURN_CODE below). GPU-heavy runs belong in the separate
# .github/workflows/cuda.yml job on a self-hosted GPU runner, keeping the
# default PR pipeline green on CPU-only workers.
if(NOT NS_ENABLE_CUDA)
    message(STATUS "NS_ENABLE_CUDA=OFF: CUDA backend tests not registered")
else()
    message(STATUS "NS_ENABLE_CUDA=ON: registering CUDA backend tests")
    find_program(NS_NVCC_EXECUTABLE nvcc)
    if(NS_NVCC_EXECUTABLE)
        message(STATUS "nvcc found: ${NS_NVCC_EXECUTABLE}")
    else()
        message(STATUS "nvcc not found in PATH; the CUDA test will self-skip")
    endif()

    add_executable(test_cuda_codegen test_cuda_codegen.cpp)
    target_link_libraries(test_cuda_codegen ns_core)
    target_compile_definitions(test_cuda_codegen PRIVATE
        NS_REPO_INCLUDE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../include"
        NS_NVCC_PATH="${NS_NVCC_EXECUTABLE}")
    add_test(NAME test_cuda_codegen COMMAND test_cuda_codegen)
    set_tests_properties(test_cuda_codegen PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 600)
endif()
