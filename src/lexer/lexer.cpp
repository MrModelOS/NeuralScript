#include "ns/lexer/lexer.hpp"
#include <cctype>
#include <stdexcept>
#include <sstream>

namespace ns {

Lexer::Lexer(const std::string& source)
    : source_(source), pos_(0), line_(1), column_(1) {}

std::vector<Token> Lexer::tokenize() {
    while (pos_ < source_.size()) {
        skip_whitespace();
        if (pos_ >= source_.size()) break;
        tokens_.push_back(next_token());
    }
    tokens_.emplace_back(TokenType::EOF_TOKEN, "", line_, column_);
    return tokens_;
}

Token Lexer::next_token() {
    skip_whitespace();
    if (pos_ >= source_.size())
        return make_token(TokenType::EOF_TOKEN);

    char c = current_char();

    if (c == '/' && peek_char() == '/') {
        skip_line_comment();
        return next_token();
    }
    if (c == '/' && peek_char() == '*') {
        skip_comment();
        return next_token();
    }

    if (std::isalpha(c) || c == '_') return read_identifier();
    if (std::isdigit(c) || (c == '.' && std::isdigit(peek_char()))) return read_number();
    if (c == '"') return read_string();

    switch (c) {
        case '+': advance(); return make_token(TokenType::OP_PLUS);
        case '-':
            advance();
            if (current_char() == '>') { advance(); return make_token(TokenType::OP_PIPELINE); }
            return make_token(TokenType::OP_MINUS);
        case '*': advance(); return make_token(TokenType::OP_STAR);
        case '/': advance(); return make_token(TokenType::OP_SLASH);
        case '%': advance(); return make_token(TokenType::OP_PERCENT);
        case '=':
            advance();
            if (current_char() == '=') { advance(); return make_token(TokenType::OP_EQ); }
            if (current_char() == '>') { advance(); return make_token(TokenType::OP_ARROW_FUNC); }
            return make_token(TokenType::OP_ASSIGN);
        case '!':
            advance();
            if (current_char() == '=') { advance(); return make_token(TokenType::OP_NEQ); }
            return make_token(TokenType::OP_NOT);
        case '<':
            advance();
            if (current_char() == '=') { advance(); return make_token(TokenType::OP_LTE); }
            return make_token(TokenType::OP_LT);
        case '>':
            advance();
            if (current_char() == '=') { advance(); return make_token(TokenType::OP_GTE); }
            return make_token(TokenType::OP_GT);
        case '&':
            advance();
            if (current_char() == '&') { advance(); return make_token(TokenType::OP_AND); }
            throw std::runtime_error("Unexpected character '&' at line " + std::to_string(line_));
        case '|':
            advance();
            if (current_char() == '|') { advance(); return make_token(TokenType::OP_OR); }
            throw std::runtime_error("Unexpected character '|' at line " + std::to_string(line_));
        case '@': advance(); return make_token(TokenType::OP_MATMUL);
        case '.': advance(); return make_token(TokenType::OP_DOT);
        case ',': advance(); return make_token(TokenType::OP_COMMA);
        case ':':
            advance();
            if (current_char() == ':') { advance(); return make_token(TokenType::OP_DOUBLE_COLON); }
            return make_token(TokenType::OP_COLON);
        case ';': advance(); return make_token(TokenType::OP_SEMICOLON);
        case '(': advance(); return make_token(TokenType::OP_LPAREN);
        case ')': advance(); return make_token(TokenType::OP_RPAREN);
        case '{': advance(); return make_token(TokenType::OP_LBRACE);
        case '}': advance(); return make_token(TokenType::OP_RBRACE);
        case '[': advance(); return make_token(TokenType::OP_LBRACKET);
        case ']': advance(); return make_token(TokenType::OP_RBRACKET);
        default: {
            std::ostringstream oss;
            oss << "Unexpected character '" << c << "' at line " << line_ << ":" << column_;
            throw std::runtime_error(oss.str());
        }
    }
}

Token Lexer::peek_token() {
    size_t saved_pos = pos_;
    uint32_t saved_line = line_;
    uint32_t saved_col = column_;
    Token tok = next_token();
    pos_ = saved_pos;
    line_ = saved_line;
    column_ = saved_col;
    return tok;
}

bool Lexer::has_more() const {
    return pos_ < source_.size();
}

char Lexer::current_char() const {
    return source_[pos_];
}

char Lexer::peek_char() const {
    if (pos_ + 1 >= source_.size()) return '\0';
    return source_[pos_ + 1];
}

void Lexer::advance() {
    if (source_[pos_] == '\n') {
        line_++;
        column_ = 1;
    } else {
        column_++;
    }
    pos_++;
}

void Lexer::skip_whitespace() {
    while (pos_ < source_.size() && std::isspace(source_[pos_])) advance();
}

void Lexer::skip_comment() {
    advance(); // skip /
    advance(); // skip *
    while (pos_ < source_.size() - 1) {
        if (source_[pos_] == '*' && source_[pos_ + 1] == '/') {
            advance(); advance();
            return;
        }
        advance();
    }
}

void Lexer::skip_line_comment() {
    advance(); advance(); // skip //
    while (pos_ < source_.size() && source_[pos_] != '\n') advance();
}

Token Lexer::read_identifier() {
    size_t start = pos_;
    uint32_t start_col = column_;
    while (pos_ < source_.size() && (std::isalnum(source_[pos_]) || source_[pos_] == '_'))
        advance();
    std::string word = source_.substr(start, pos_ - start);
    TokenType type = keyword_or_identifier(word);
    return Token(type, word, line_, start_col);
}

Token Lexer::read_number() {
    size_t start = pos_;
    uint32_t start_col = column_;
    bool is_float = false;

    while (pos_ < source_.size() && std::isdigit(source_[pos_])) advance();

    if (pos_ < source_.size() && source_[pos_] == '.' &&
        pos_ + 1 < source_.size() && std::isdigit(source_[pos_ + 1])) {
        is_float = true;
        advance(); // skip .
        while (pos_ < source_.size() && std::isdigit(source_[pos_])) advance();
    }

    if (pos_ < source_.size() && (source_[pos_] == 'e' || source_[pos_] == 'E')) {
        is_float = true;
        advance();
        if (pos_ < source_.size() && (source_[pos_] == '+' || source_[pos_] == '-')) advance();
        while (pos_ < source_.size() && std::isdigit(source_[pos_])) advance();
    }

    std::string value = source_.substr(start, pos_ - start);
    return Token(is_float ? TokenType::FLOAT_LITERAL : TokenType::INT_LITERAL,
                 value, line_, start_col);
}

Token Lexer::read_string() {
    uint32_t start_col = column_;
    advance(); // skip opening "
    std::string value;
    while (pos_ < source_.size() && source_[pos_] != '"') {
        if (source_[pos_] == '\\') {
            advance();
            if (pos_ < source_.size()) {
                switch (source_[pos_]) {
                    case 'n': value += '\n'; break;
                    case 't': value += '\t'; break;
                    case '\\': value += '\\'; break;
                    case '"': value += '"'; break;
                    default: value += source_[pos_]; break;
                }
            }
        } else {
            value += source_[pos_];
        }
        advance();
    }
    if (pos_ < source_.size()) advance(); // skip closing "
    return Token(TokenType::STRING_LITERAL, value, line_, start_col);
}

Token Lexer::make_token(TokenType type, const std::string& value) {
    return Token(type, value, line_, column_ - (value.empty() ? 0 : value.size()));
}

TokenType Lexer::keyword_or_identifier(const std::string& word) {
    // Keywords
    if (word == "fn") return TokenType::KW_FN;
    if (word == "var") return TokenType::KW_VAR;
    if (word == "return") return TokenType::KW_RETURN;
    if (word == "if") return TokenType::KW_IF;
    if (word == "else") return TokenType::KW_ELSE;
    if (word == "while") return TokenType::KW_WHILE;
    if (word == "for") return TokenType::KW_FOR;
    if (word == "network") return TokenType::KW_NETWORK;
    if (word == "layer") return TokenType::KW_LAYER;
    if (word == "forward") return TokenType::KW_FORWARD;
    if (word == "train") return TokenType::KW_TRAIN;
    if (word == "grad") return TokenType::KW_GRAD;
    if (word == "mut") return TokenType::KW_MUT;
    if (word == "ref") return TokenType::KW_REF;
    if (word == "type") return TokenType::KW_TYPE;
    if (word == "as") return TokenType::KW_AS;

    // Type keywords
    if (word == "int") return TokenType::TYPE_INT;
    if (word == "float") return TokenType::TYPE_FLOAT;
    if (word == "bool") return TokenType::TYPE_BOOL;
    if (word == "string") return TokenType::TYPE_STRING;
    if (word == "Tensor") return TokenType::TYPE_TENSOR;
    if (word == "Dynamic") return TokenType::TYPE_DYNAMIC;

    // Dtype keywords
    if (word == "float16") return TokenType::DTYPE_FLOAT16;
    if (word == "float32") return TokenType::DTYPE_FLOAT32;
    if (word == "float64") return TokenType::DTYPE_FLOAT64;
    if (word == "int8") return TokenType::DTYPE_INT8;
    if (word == "int16") return TokenType::DTYPE_INT16;
    if (word == "int32") return TokenType::DTYPE_INT32;
    if (word == "int64") return TokenType::DTYPE_INT64;
    if (word == "fp8") return TokenType::DTYPE_FP8;
    if (word == "fp4") return TokenType::DTYPE_FP4;

    // Activation functions
    if (word == "ReLU") return TokenType::ACT_RELU;
    if (word == "LeakyReLU") return TokenType::ACT_LEAKY_RELU;
    if (word == "Sigmoid") return TokenType::ACT_SIGMOID;
    if (word == "Tanh") return TokenType::ACT_TANH;
    if (word == "Swish") return TokenType::ACT_SWISH;
    if (word == "GELU") return TokenType::ACT_GELU;
    if (word == "SwiGLU") return TokenType::ACT_SWIGLU;
    if (word == "SiLU") return TokenType::ACT_SILU;
    if (word == "Identity") return TokenType::ACT_IDENTITY;
    if (word == "Softmax") return TokenType::ACT_SOFTMAX;

    // Optimizer keywords
    if (word == "AdamW") return TokenType::OPT_ADAMW;
    if (word == "Muon") return TokenType::OPT_MUON;
    if (word == "SGD") return TokenType::OPT_SGD;

    // Bool literals
    if (word == "true" || word == "false") return TokenType::IDENTIFIER;

    return TokenType::IDENTIFIER;
}

} // namespace ns
