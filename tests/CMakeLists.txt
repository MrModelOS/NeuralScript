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

add_executable(test_autodiff test_autodiff.cpp)
add_test(NAME test_autodiff COMMAND test_autodiff)

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
