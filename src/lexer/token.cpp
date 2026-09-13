#include "ns/lexer/token.hpp"

namespace ns {

const char* token_type_name(TokenType type) {
    switch (type) {
        case TokenType::INT_LITERAL: return "INT_LITERAL";
        case TokenType::FLOAT_LITERAL: return "FLOAT_LITERAL";
        case TokenType::STRING_LITERAL: return "STRING_LITERAL";
        case TokenType::IDENTIFIER: return "IDENTIFIER";
        case TokenType::TYPE_INT: return "int";
        case TokenType::TYPE_FLOAT: return "float";
        case TokenType::TYPE_BOOL: return "bool";
        case TokenType::TYPE_STRING: return "string";
        case TokenType::TYPE_TENSOR: return "Tensor";
        case TokenType::TYPE_DYNAMIC: return "Dynamic";
        case TokenType::KW_FN: return "fn";
        case TokenType::KW_VAR: return "var";
        case TokenType::KW_RETURN: return "return";
        case TokenType::KW_IF: return "if";
        case TokenType::KW_ELSE: return "else";
        case TokenType::KW_WHILE: return "while";
        case TokenType::KW_FOR: return "for";
        case TokenType::KW_NETWORK: return "network";
        case TokenType::KW_LAYER: return "layer";
        case TokenType::KW_FORWARD: return "forward";
        case TokenType::KW_INPUT: return "input";
        case TokenType::KW_OUTPUT: return "output";
        case TokenType::KW_GRAD: return "grad";
        case TokenType::KW_MUT: return "mut";
        case TokenType::KW_REF: return "ref";
        case TokenType::KW_TYPE: return "type";
        case TokenType::KW_AS: return "as";
        case TokenType::KW_TRAIN: return "train";
        case TokenType::DTYPE_FLOAT16: return "float16";
        case TokenType::DTYPE_FLOAT32: return "float32";
        case TokenType::DTYPE_FLOAT64: return "float64";
        case TokenType::DTYPE_INT8: return "int8";
        case TokenType::DTYPE_INT16: return "int16";
        case TokenType::DTYPE_INT32: return "int32";
        case TokenType::DTYPE_INT64: return "int64";
        case TokenType::DTYPE_FP8: return "fp8";
        case TokenType::DTYPE_FP4: return "fp4";
        case TokenType::DTYPE_BOOL: return "bool";
        case TokenType::OP_PLUS: return "+";
        case TokenType::OP_MINUS: return "-";
        case TokenType::OP_STAR: return "*";
        case TokenType::OP_SLASH: return "/";
        case TokenType::OP_PERCENT: return "%";
        case TokenType::OP_ASSIGN: return "=";
        case TokenType::OP_EQ: return "==";
        case TokenType::OP_NEQ: return "!=";
        case TokenType::OP_LT: return "<";
        case TokenType::OP_GT: return ">";
        case TokenType::OP_LTE: return "<=";
        case TokenType::OP_GTE: return ">=";
        case TokenType::OP_AND: return "&&";
        case TokenType::OP_OR: return "||";
        case TokenType::OP_NOT: return "!";
        case TokenType::OP_MATMUL: return "@";
        case TokenType::OP_PIPELINE: return "->";
        case TokenType::OP_ARROW_FUNC: return "=>";
        case TokenType::OP_DOT: return ".";
        case TokenType::OP_COMMA: return ",";
        case TokenType::OP_COLON: return ":";
        case TokenType::OP_SEMICOLON: return ";";
        case TokenType::OP_LPAREN: return "(";
        case TokenType::OP_RPAREN: return ")";
        case TokenType::OP_LBRACE: return "{";
        case TokenType::OP_RBRACE: return "}";
        case TokenType::OP_LBRACKET: return "[";
        case TokenType::OP_RBRACKET: return "]";
        case TokenType::OP_DOUBLE_COLON: return "::";
        case TokenType::ACT_RELU: return "ReLU";
        case TokenType::ACT_LEAKY_RELU: return "LeakyReLU";
        case TokenType::ACT_SIGMOID: return "Sigmoid";
        case TokenType::ACT_TANH: return "Tanh";
        case TokenType::ACT_SWISH: return "Swish";
        case TokenType::ACT_GELU: return "GELU";
        case TokenType::ACT_SWIGLU: return "SwiGLU";
        case TokenType::ACT_SILU: return "SiLU";
        case TokenType::ACT_IDENTITY: return "Identity";
        case TokenType::ACT_SOFTMAX: return "Softmax";
        case TokenType::OPT_ADAMW: return "AdamW";
        case TokenType::OPT_MUON: return "Muon";
        case TokenType::OPT_SGD: return "SGD";
        case TokenType::EOF_TOKEN: return "EOF";
        case TokenType::ERROR: return "ERROR";
    }
    return "UNKNOWN";
}

} // namespace ns
