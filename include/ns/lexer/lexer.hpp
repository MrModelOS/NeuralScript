#pragma once
#include "ns/lexer/token.hpp"
#include <string>
#include <vector>

namespace ns {

class Lexer {
public:
    explicit Lexer(const std::string& source);

    std::vector<Token> tokenize();
    Token next_token();
    Token peek_token();
    bool has_more() const;

private:
    std::string source_;
    size_t pos_;
    uint32_t line_;
    uint32_t column_;
    std::vector<Token> tokens_;

    char current_char() const;
    char peek_char() const;
    void advance();
    void skip_whitespace();
    void skip_comment();
    void skip_line_comment();

    Token read_identifier();
    Token read_number();
    Token read_string();
    Token make_token(TokenType type, const std::string& value = "");

    TokenType keyword_or_identifier(const std::string& word);
};

} // namespace ns
