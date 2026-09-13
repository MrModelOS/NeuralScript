#include "ns/parser/parser.hpp"
#include <stdexcept>
#include <sstream>

namespace ns {

Parser::Parser(const std::vector<Token>& tokens) : tokens_(tokens), current_(0) {}

// ---- Token helpers ----

const Token& Parser::peek() const {
    if (current_ >= tokens_.size()) return tokens_.back();
    return tokens_[current_];
}

const Token& Parser::peek_ahead(size_t n) const {
    size_t idx = std::min(current_ + n, tokens_.size() - 1);
    return tokens_[idx];
}

const Token& Parser::previous() const {
    return tokens_[current_ - 1];
}

const Token& Parser::advance() {
    if (current_ < tokens_.size()) current_++;
    return tokens_[current_ - 1];
}

bool Parser::check(TokenType type) const {
    return peek().type == type;
}

bool Parser::match(TokenType type) {
    if (check(type)) {
        advance();
        return true;
    }
    return false;
}

bool Parser::match_one_of(std::initializer_list<TokenType> types) {
    for (auto t : types) {
        if (match(t)) return true;
    }
    return false;
}

const Token& Parser::expect(TokenType type, const std::string& msg) {
    if (check(type)) return advance();
    std::ostringstream oss;
    oss << "Expected " << token_type_name(type) << " but got " << token_type_name(peek().type)
        << " '" << peek().value << "' at " << peek().line << ":" << peek().column << ": " << msg;
    throw std::runtime_error(oss.str());
}

void Parser::error(const Token& token, const std::string& msg) {
    std::ostringstream oss;
    oss << "Parse error at " << token.line << ":" << token.column << ": " << msg;
    throw std::runtime_error(oss.str());
}

bool Parser::at_end() const {
    return peek().type == TokenType::EOF_TOKEN;
}

// ---- Program ----

Program Parser::parse_program() {
    Program program;
    while (!at_end()) {
        if (match(TokenType::OP_SEMICOLON)) continue;
        program.add(parse_statement());
    }
    return program;
}

// ---- Statements ----

StmtPtr Parser::parse_statement() {
    Token tok = peek();

    switch (tok.type) {
        case TokenType::KW_VAR: {
            Token var_tok = advance();
            return parse_var_decl(var_tok);
        }
        case TokenType::KW_FN: {
            Token fn_tok = advance();
            return parse_fn_decl(fn_tok);
        }
        case TokenType::KW_NETWORK: {
            Token net_tok = advance();
            return parse_network_decl(net_tok);
        }
        case TokenType::KW_GRAD: {
            Token grad_tok = advance();
            return parse_grad_block(grad_tok);
        }
        case TokenType::KW_RETURN: {
            return parse_return_stmt();
        }
        case TokenType::KW_IF: {
            return parse_if_stmt();
        }
        case TokenType::KW_WHILE: {
            return parse_while_stmt();
        }
        case TokenType::KW_TYPE: {
            Token type_tok = advance();
            return parse_type_decl(type_tok);
        }
        case TokenType::KW_LAYER: {
            Token layer_tok = advance();
            return parse_layer_decl(layer_tok);
        }
        case TokenType::OP_LBRACE: {
            return parse_block();
        }
        case TokenType::KW_FORWARD: {
            // forward(x) { return x -> fc1 -> ... }
            Token fwd_tok = advance();
            auto stmt = std::make_unique<Stmt>(Stmt::FORWARD_DECL, fwd_tok);
            expect(TokenType::OP_LPAREN, "expected ( after forward");
            if (!check(TokenType::OP_RPAREN)) {
                while (!check(TokenType::OP_RPAREN)) {
                    Token p = advance();
                    stmt->params.push_back({p.value, nullptr, false, false});
                    if (!match(TokenType::OP_COMMA)) break;
                }
            }
            expect(TokenType::OP_RPAREN, "expected ) after forward params");
            stmt->body = parse_block();
            return stmt;
        }
        case TokenType::KW_TRAIN: {
            // train(x, y) -> loss_type { grad { ... } }
            return parse_train_decl(advance());
        }
        default:
            return parse_expr_stmt();
    }
}

StmtPtr Parser::parse_var_decl(Token tok) {
    auto stmt = std::make_unique<Stmt>(Stmt::VAR_DECL, tok);
    Token name = expect(TokenType::IDENTIFIER, "Expected variable name");
    stmt->var_name = name.value;

    if (match(TokenType::OP_COLON)) {
        stmt->var_type = parse_type();
    }

    if (match(TokenType::OP_ASSIGN)) {
        stmt->init_expr = parse_expression();
    }

    match(TokenType::OP_SEMICOLON);
    return stmt;
}

StmtPtr Parser::parse_fn_decl(Token tok) {
    auto stmt = std::make_unique<Stmt>(Stmt::FN_DECL, tok);
    Token name = expect(TokenType::IDENTIFIER, "Expected function name");
    stmt->fn_name = name.value;

    expect(TokenType::OP_LPAREN, "expected ( after function name");
    while (!check(TokenType::OP_RPAREN)) {
        Stmt::Param param;
        param.is_mut = match(TokenType::KW_MUT);
        param.is_ref = match(TokenType::KW_REF);
        Token pname = expect(TokenType::IDENTIFIER, "Expected parameter name");
        param.name = pname.value;
        if (match(TokenType::OP_COLON)) {
            param.type = parse_type();
        }
        stmt->params.push_back(std::move(param));
        if (!match(TokenType::OP_COMMA)) break;
    }
    expect(TokenType::OP_RPAREN, "expected ) after params");

    // Return type: "-> Type" or ": Type"
    if (match(TokenType::OP_PIPELINE) || match(TokenType::OP_COLON)) {
        stmt->return_type = parse_type();
    }

    stmt->body = parse_block();
    return stmt;
}

StmtPtr Parser::parse_train_decl(Token tok) {
    // train(x: Tensor[Batch, F], y: Tensor[Batch, C]) -> float32 { grad { ... } }
    auto stmt = std::make_unique<Stmt>(Stmt::TRAIN_DECL, tok);

    expect(TokenType::OP_LPAREN, "expected ( after train");
    while (!check(TokenType::OP_RPAREN)) {
        Stmt::Param param;
        param.is_mut = match(TokenType::KW_MUT);
        param.is_ref = match(TokenType::KW_REF);
        Token pname = expect(TokenType::IDENTIFIER, "Expected parameter name");
        param.name = pname.value;
        if (match(TokenType::OP_COLON)) {
            param.type = parse_type();
        }
        stmt->params.push_back(std::move(param));
        if (!match(TokenType::OP_COMMA)) break;
    }
    expect(TokenType::OP_RPAREN, "expected ) after train params");

    if (match(TokenType::OP_PIPELINE) || match(TokenType::OP_COLON)) {
        stmt->return_type = parse_type();
    }

    stmt->body = parse_block();
    return stmt;
}

StmtPtr Parser::parse_network_decl(Token tok) {
    auto stmt = std::make_unique<Stmt>(Stmt::NETWORK_DECL, tok);

    Token name = expect(TokenType::IDENTIFIER, "Expected network name");
    stmt->network_name = name.value;

    expect(TokenType::OP_LBRACE, "expected { after network name");

    while (!check(TokenType::OP_RBRACE)) {
        // Detect input/output sections: <name> ':' Tensor[...]  (or <name> ':')
        if (check(TokenType::IDENTIFIER) &&
            peek_ahead(1).type == TokenType::OP_COLON &&
            peek_ahead(2).type == TokenType::TYPE_TENSOR) {
            Token io = advance();
            advance(); // consume ':'
            auto sub = std::make_unique<Stmt>(Stmt::VAR_DECL, io);
            sub->var_name = io.value;
            sub->var_type = parse_type();
            match(TokenType::OP_SEMICOLON);
            stmt->methods.push_back(std::move(sub));
        }
        else if (check(TokenType::KW_LAYER)) {
            advance();
            stmt->layers.push_back(parse_layer_decl(previous()));
        }
        else if (check(TokenType::KW_FORWARD)) {
            stmt->methods.push_back(parse_statement());
        }
        else if (check(TokenType::KW_TRAIN)) {
            stmt->methods.push_back(parse_statement());
        }
        else if (check(TokenType::KW_TYPE)) {
            advance();
            stmt->methods.push_back(parse_type_decl(previous()));
        }
        else {
            // treat as general statement
            stmt->methods.push_back(parse_statement());
        }
    }
    expect(TokenType::OP_RBRACE, "expected } after network");

    return stmt;
}

StmtPtr Parser::parse_layer_decl(Token tok) {
    // layer fc1 = Dense(in: InDim, out: 512, activation: ReLU)
    auto stmt = std::make_unique<Stmt>(Stmt::LAYER_DECL, tok);

    Token name = expect(TokenType::IDENTIFIER, "Expected layer name");
    stmt->layer_name = name.value;

    expect(TokenType::OP_ASSIGN, "expected '=' after layer name");

    Token layer_type = expect(TokenType::IDENTIFIER, "Expected layer type (Dense, Dropout, etc.)");
    stmt->layer_type = layer_type.value;

    if (match(TokenType::OP_LPAREN)) {
        while (!check(TokenType::OP_RPAREN)) {
            Stmt::LayerParam lp;
            Token lp_name = expect(TokenType::IDENTIFIER, "Expected parameter name");
            lp.name = lp_name.value;
            expect(TokenType::OP_COLON, "expected ':' after layer param name");
            lp.value = parse_expression();
            stmt->layer_params.push_back(std::move(lp));
            if (!match(TokenType::OP_COMMA)) break;
        }
        expect(TokenType::OP_RPAREN, "expected ) after layer params");
    }

    match(TokenType::OP_SEMICOLON);
    return stmt;
}

StmtPtr Parser::parse_grad_block(Token tok) {
    auto stmt = std::make_unique<Stmt>(Stmt::GRAD_BLOCK, tok);
    if (match(TokenType::OP_LPAREN)) {
        while (!check(TokenType::OP_RPAREN)) {
            Token p = advance();
            stmt->params.push_back({p.value, nullptr, false, false});
            if (!match(TokenType::OP_COMMA)) break;
        }
        expect(TokenType::OP_RPAREN, "expected ) after grad params");
    }
    stmt->grad_body = parse_block();
    return stmt;
}

StmtPtr Parser::parse_block() {
    Token tok = expect(TokenType::OP_LBRACE, "expected { to start block");
    auto stmt = std::make_unique<Stmt>(Stmt::BLOCK, tok);
    while (!check(TokenType::OP_RBRACE) && !at_end()) {
        stmt->statements.push_back(parse_statement());
    }
    expect(TokenType::OP_RBRACE, "expected } to close block");
    return stmt;
}

StmtPtr Parser::parse_return_stmt() {
    Token tok = advance();
    auto stmt = std::make_unique<Stmt>(Stmt::RETURN_STMT, tok);
    if (!check(TokenType::OP_SEMICOLON) && !check(TokenType::OP_RBRACE)) {
        stmt->init_expr = parse_expression();
    }
    match(TokenType::OP_SEMICOLON);
    return stmt;
}

StmtPtr Parser::parse_if_stmt() {
    Token tok = advance();
    auto stmt = std::make_unique<Stmt>(Stmt::IF_STMT, tok);
    stmt->condition = parse_expression();
    stmt->body = parse_statement();
    if (match(TokenType::KW_ELSE)) {
        stmt->else_branch = parse_statement();
    }
    return stmt;
}

StmtPtr Parser::parse_while_stmt() {
    Token tok = advance();
    auto stmt = std::make_unique<Stmt>(Stmt::WHILE_STMT, tok);
    stmt->condition = parse_expression();
    stmt->body = parse_statement();
    return stmt;
}

StmtPtr Parser::parse_expr_stmt() {
    Token tok = peek();
    auto stmt = std::make_unique<Stmt>(Stmt::EXPR_STMT, tok);
    stmt->expr = parse_expression();
    match(TokenType::OP_SEMICOLON);
    return stmt;
}

StmtPtr Parser::parse_type_decl(Token tok) {
    // type Batch = Dynamic
    // type Features = 784
    auto stmt = std::make_unique<Stmt>(Stmt::TYPE_DECL, tok);
    Token name = expect(TokenType::IDENTIFIER, "Expected type alias name");
    stmt->alias_name = name.value;
    expect(TokenType::OP_ASSIGN, "expected '=' in type declaration");
    stmt->alias_expr = parse_primary();
    match(TokenType::OP_SEMICOLON);
    return stmt;
}

// ---- Expressions ----

ExprPtr Parser::parse_expression() {
    return parse_assignment();
}

ExprPtr Parser::parse_assignment() {
    ExprPtr expr = parse_pipeline();
    if (match(TokenType::OP_ASSIGN)) {
        auto assign = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        assign->left = std::move(expr);
        assign->right = parse_assignment();
        return assign;
    }
    return expr;
}

ExprPtr Parser::parse_pipeline() {
    ExprPtr left = parse_or();
    while (match(TokenType::OP_PIPELINE)) {
        auto pip = std::make_unique<Expr>(Expr::PIPELINE_OP, previous());
        pip->left = std::move(left);
        pip->right = parse_or();
        left = std::move(pip);
    }
    return left;
}

ExprPtr Parser::parse_or() {
    ExprPtr left = parse_and();
    while (match(TokenType::OP_OR)) {
        auto expr = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_and();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_and() {
    ExprPtr left = parse_equality();
    while (match(TokenType::OP_AND)) {
        auto expr = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_equality();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_equality() {
    ExprPtr left = parse_comparison();
    while (match_one_of({TokenType::OP_EQ, TokenType::OP_NEQ})) {
        auto expr = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_comparison();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_comparison() {
    ExprPtr left = parse_additive();
    while (match_one_of({TokenType::OP_LT, TokenType::OP_GT, TokenType::OP_LTE, TokenType::OP_GTE})) {
        auto expr = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_additive();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_additive() {
    ExprPtr left = parse_multiplicative();
    while (match_one_of({TokenType::OP_PLUS, TokenType::OP_MINUS})) {
        auto expr = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_multiplicative();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_multiplicative() {
    ExprPtr left = parse_matmul();
    while (match_one_of({TokenType::OP_STAR, TokenType::OP_SLASH, TokenType::OP_PERCENT})) {
        auto expr = std::make_unique<Expr>(Expr::BINARY_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_matmul();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_matmul() {
    ExprPtr left = parse_unary();
    while (match(TokenType::OP_MATMUL)) {
        auto expr = std::make_unique<Expr>(Expr::MATMUL_OP, previous());
        expr->left = std::move(left);
        expr->right = parse_unary();
        left = std::move(expr);
    }
    return left;
}

ExprPtr Parser::parse_unary() {
    if (match_one_of({TokenType::OP_NOT, TokenType::OP_MINUS})) {
        auto expr = std::make_unique<Expr>(Expr::UNARY_OP, previous());
        expr->operand = parse_unary();
        return expr;
    }
    return parse_postfix();
}

ExprPtr Parser::parse_postfix() {
    ExprPtr expr = parse_primary();

    while (true) {
        if (match(TokenType::OP_LPAREN)) {
            expr = parse_function_call(std::move(expr));
        } else if (match(TokenType::OP_LBRACKET)) {
            auto index_expr = std::make_unique<Expr>(Expr::INDEX_OP, previous());
            index_expr->operand = std::move(expr);
            if (!check(TokenType::OP_RBRACKET)) {
                do {
                    index_expr->indices.push_back(parse_expression());
                } while (match(TokenType::OP_COMMA));
            }
            expect(TokenType::OP_RBRACKET, "expected ] after index");
            expr = std::move(index_expr);
        } else if (match(TokenType::OP_DOT)) {
            // Member access or method call: model.forward(...) / opt.step(model)
            if (!check(TokenType::IDENTIFIER) && !check(TokenType::KW_FORWARD) &&
                !check(TokenType::KW_TRAIN)) {
                error(peek(), "expected member name after '.'");
            }
            Token member = advance();
            auto member_expr = std::make_unique<Expr>(Expr::IDENTIFIER, member);
            if (match(TokenType::OP_LPAREN)) {
                auto call = std::make_unique<Expr>(Expr::FUNCTION_CALL, member);
                call->operand = std::move(member_expr);
                if (!check(TokenType::OP_RPAREN)) {
                    do {
                        call->args.push_back(parse_expression());
                    } while (match(TokenType::OP_COMMA));
                }
                expect(TokenType::OP_RPAREN, "expected ) after method arguments");
                expr = std::move(call);
            } else {
                expr = std::move(member_expr);
            }
        } else {
            break;
        }
    }
    return expr;
}

ExprPtr Parser::parse_primary() {
    Token tok = peek();

    switch (tok.type) {
        case TokenType::INT_LITERAL: {
            advance();
            return std::make_unique<Expr>(Expr::LITERAL_INT, tok);
        }
        case TokenType::FLOAT_LITERAL: {
            advance();
            return std::make_unique<Expr>(Expr::LITERAL_FLOAT, tok);
        }
        case TokenType::STRING_LITERAL: {
            advance();
            return std::make_unique<Expr>(Expr::LITERAL_STRING, tok);
        }
        case TokenType::IDENTIFIER: {
            advance();
            // bool literals
            if (tok.value == "true" || tok.value == "false") {
                return std::make_unique<Expr>(Expr::LITERAL_BOOL, tok);
            }
            return std::make_unique<Expr>(Expr::IDENTIFIER, tok);
        }
        case TokenType::OP_LPAREN: {
            advance();
            ExprPtr expr = parse_expression();
            expect(TokenType::OP_RPAREN, "expected ) after expression");
            return expr;
        }
        case TokenType::OP_MINUS: {
            return parse_unary();
        }
        case TokenType::TYPE_DYNAMIC: {
            advance();
            return std::make_unique<Expr>(Expr::IDENTIFIER, tok);
        }
        case TokenType::ACT_RELU:
        case TokenType::ACT_LEAKY_RELU:
        case TokenType::ACT_SIGMOID:
        case TokenType::ACT_TANH:
        case TokenType::ACT_SWISH:
        case TokenType::ACT_GELU:
        case TokenType::ACT_SWIGLU:
        case TokenType::ACT_SILU:
        case TokenType::ACT_IDENTITY:
        case TokenType::ACT_SOFTMAX: {
            advance();
            return std::make_unique<Expr>(Expr::IDENTIFIER, tok);
        }
        case TokenType::OPT_ADAMW:
        case TokenType::OPT_MUON:
        case TokenType::OPT_SGD: {
            advance();
            return std::make_unique<Expr>(Expr::IDENTIFIER, tok);
        }
        default: {
            error(tok, "Unexpected token in expression");
        }
    }
    return nullptr;
}

ExprPtr Parser::parse_function_call(ExprPtr callee) {
    auto expr = std::make_unique<Expr>(Expr::FUNCTION_CALL, previous());
    expr->operand = std::move(callee);

    if (!check(TokenType::OP_RPAREN)) {
        do {
            expr->args.push_back(parse_expression());
        } while (match(TokenType::OP_COMMA));
    }
    expect(TokenType::OP_RPAREN, "expected ) after function arguments");
    return expr;
}

// ---- Types ----

TypePtr Parser::parse_type() {
    if (check(TokenType::TYPE_TENSOR) || check(TokenType::TYPE_DYNAMIC)) {
        if (check(TokenType::TYPE_TENSOR)) {
            return std::make_unique<TypeNode>(parse_tensor_type());
        }
    }
    if (is_dtype(peek().type)) {
        Dtype dt = token_to_dtype(advance().type);
        return std::make_unique<TypeNode>(dt);
    }
    if (check(TokenType::TYPE_INT) || check(TokenType::TYPE_FLOAT) ||
        check(TokenType::TYPE_BOOL) || check(TokenType::TYPE_STRING)) {
        Dtype dt = Dtype::Int64; // default mapping
        switch (advance().type) {
            case TokenType::TYPE_INT: dt = Dtype::Int64; break;
            case TokenType::TYPE_FLOAT: dt = Dtype::Float64; break;
            case TokenType::TYPE_BOOL: dt = Dtype::Bool; break;
            default: dt = Dtype::Int64;
        }
        return std::make_unique<TypeNode>(dt);
    }
    Token tok = peek();
    error(tok, "Expected type");
    return nullptr;
}

TensorType Parser::parse_tensor_type() {
    Token tensor_tok = expect(TokenType::TYPE_TENSOR, "expected Tensor");
    (void)tensor_tok;
    expect(TokenType::OP_LBRACKET, "expected [ after Tensor");

    std::vector<DimExpr> dims;
    do {
        if (check(TokenType::TYPE_DYNAMIC) || check(TokenType::IDENTIFIER)) {
            Token t = advance();
            if (t.value == "Dynamic") {
                dims.push_back(DimExpr::dynamic());
            } else {
                dims.push_back(DimExpr::symbolic(t.value));
            }
        } else if (check(TokenType::INT_LITERAL)) {
            Token t = advance();
            dims.push_back(DimExpr::constant(std::stoll(t.value)));
        } else {
            error(peek(), "Expected dimension (constant, identifier, or Dynamic)");
        }
    } while (match(TokenType::OP_COMMA));

    expect(TokenType::OP_RBRACKET, "expected ] after tensor dims");

    Dtype dtype = Dtype::Float32;
    if (is_dtype(peek().type)) {
        dtype = token_to_dtype(advance().type);
    }

    return TensorType(std::move(dims), dtype);
}

Dtype Parser::parse_dtype() {
    if (!is_dtype(peek().type)) {
        error(peek(), "Expected dtype");
    }
    return token_to_dtype(advance().type);
}

bool Parser::is_dtype(TokenType tt) const {
    switch (tt) {
        case TokenType::DTYPE_FLOAT16:
        case TokenType::DTYPE_FLOAT32:
        case TokenType::DTYPE_FLOAT64:
        case TokenType::DTYPE_INT8:
        case TokenType::DTYPE_INT16:
        case TokenType::DTYPE_INT32:
        case TokenType::DTYPE_INT64:
        case TokenType::DTYPE_FP8:
        case TokenType::DTYPE_FP4:
        case TokenType::DTYPE_BOOL:
            return true;
        default:
            return false;
    }
}

} // namespace ns
