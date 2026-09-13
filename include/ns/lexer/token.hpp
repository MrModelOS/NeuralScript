#pragma once
#include <string>
#include <cstdint>

namespace ns {

enum class TokenType {
    // Literals
    INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL, IDENTIFIER,

    // Types
    TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING,
    TYPE_TENSOR, TYPE_DYNAMIC,

    // Keywords
    KW_FN, KW_VAR, KW_RETURN, KW_IF, KW_ELSE, KW_WHILE, KW_FOR,
    KW_NETWORK, KW_LAYER, KW_FORWARD, KW_INPUT, KW_OUTPUT,
    KW_GRAD, KW_MUT, KW_REF, KW_TYPE, KW_AS, KW_TRAIN,

    // Dtypes
    DTYPE_FLOAT16, DTYPE_FLOAT32, DTYPE_FLOAT64, DTYPE_INT8, DTYPE_INT16,
    DTYPE_INT32, DTYPE_INT64, DTYPE_FP8, DTYPE_FP4, DTYPE_BOOL,

    // Operators
    OP_PLUS, OP_MINUS, OP_STAR, OP_SLASH, OP_PERCENT,
    OP_ASSIGN, OP_EQ, OP_NEQ, OP_LT, OP_GT, OP_LTE, OP_GTE,
    OP_AND, OP_OR, OP_NOT,
    OP_MATMUL,       // @
    OP_PIPELINE,     // ->
    OP_ARROW_FUNC,   // =>
    OP_DOT, OP_COMMA, OP_COLON, OP_SEMICOLON,
    OP_LPAREN, OP_RPAREN, OP_LBRACE, OP_RBRACE,
    OP_LBRACKET, OP_RBRACKET,
    OP_DOUBLE_COLON, // ::

    // Activation functions
    ACT_RELU, ACT_LEAKY_RELU, ACT_SIGMOID, ACT_TANH, ACT_SWISH,
    ACT_GELU, ACT_SWIGLU, ACT_SILU, ACT_IDENTITY, ACT_SOFTMAX,

    // Optimizer keywords
    OPT_ADAMW, OPT_MUON, OPT_SGD,

    // Special
    EOF_TOKEN, ERROR
};

struct Token {
    TokenType type;
    std::string value;
    uint32_t line;
    uint32_t column;

    Token() : type(TokenType::ERROR), line(0), column(0) {}
    Token(TokenType t, const std::string& v, uint32_t l, uint32_t c)
        : type(t), value(v), line(l), column(c) {}

    bool is(TokenType t) const { return type == t; }
    bool is_one_of(std::initializer_list<TokenType> types) const {
        for (auto t : types) if (type == t) return true;
        return false;
    }
};

const char* token_type_name(TokenType type);

} // namespace ns
