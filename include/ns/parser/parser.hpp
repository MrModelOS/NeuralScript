#pragma once
#include "ns/parser/ast.hpp"
#include "ns/lexer/token.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ns {

class Parser {
public:
    explicit Parser(const std::vector<Token>& tokens);
    Program parse_program();

private:
    const std::vector<Token>& tokens_;
    size_t current_;

    // Token helpers
    const Token& peek() const;
    const Token& peek_ahead(size_t n) const;
    const Token& previous() const;
    const Token& advance();
    bool check(TokenType type) const;
    bool match(TokenType type);
    bool match_one_of(std::initializer_list<TokenType> types);
    const Token& expect(TokenType type, const std::string& msg);
    void error(const Token& token, const std::string& msg);
    bool at_end() const;

    // Statement parsing
    StmtPtr parse_statement();
    StmtPtr parse_var_decl(Token tok);
    StmtPtr parse_fn_decl(Token tok);
    StmtPtr parse_network_decl(Token tok);
    StmtPtr parse_grad_block(Token tok);
    StmtPtr parse_block();
    StmtPtr parse_return_stmt();
    StmtPtr parse_if_stmt();
    StmtPtr parse_while_stmt();
    StmtPtr parse_expr_stmt();
    StmtPtr parse_type_decl(Token tok);
    StmtPtr parse_layer_decl(Token tok);

    // Expression parsing
    ExprPtr parse_expression();
    ExprPtr parse_assignment();
    ExprPtr parse_pipeline();
    ExprPtr parse_or();
    ExprPtr parse_and();
    ExprPtr parse_equality();
    ExprPtr parse_comparison();
    ExprPtr parse_additive();
    ExprPtr parse_multiplicative();
    ExprPtr parse_matmul();
    ExprPtr parse_unary();
    ExprPtr parse_postfix();
    ExprPtr parse_primary();
    ExprPtr parse_function_call(ExprPtr callee);
    ExprPtr parse_index_op(ExprPtr base);

    // Type parsing
    TypePtr parse_type();
    TensorType parse_tensor_type();
    Dtype parse_dtype();

    // Helper
    bool is_dtype(TokenType tt) const;
};

} // namespace ns
