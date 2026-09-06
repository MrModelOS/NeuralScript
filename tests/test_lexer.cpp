#include "ns/lexer/lexer.hpp"
#include <cassert>
#include <iostream>

using namespace ns;

int main() {
    // Basic tokenization
    {
        Lexer lexer("var x: Tensor[784] float32");
        auto toks = lexer.tokenize();
        assert(toks[0].type == TokenType::KW_VAR);
        assert(toks[1].type == TokenType::IDENTIFIER);
        assert(toks[1].value == "x");
        assert(toks[2].type == TokenType::OP_COLON);
        assert(toks[3].type == TokenType::TYPE_TENSOR);
        assert(toks[4].type == TokenType::OP_LBRACKET);
        assert(toks[5].type == TokenType::INT_LITERAL);
        assert(toks[5].value == "784");
        assert(toks[7].type == TokenType::DTYPE_FLOAT32);
    }

    // Pipeline operator and matmul
    {
        Lexer lexer("a @ b -> c");
        auto toks = lexer.tokenize();
        assert(toks[1].type == TokenType::OP_MATMUL);
        assert(toks[3].type == TokenType::OP_PIPELINE);
    }

    // Comments
    {
        Lexer lexer("// line comment\nvar x\n/* block */ var y");
        auto toks = lexer.tokenize();
        assert(toks[0].type == TokenType::KW_VAR);
        assert(toks[2].type == TokenType::KW_VAR);
    }

    // Activation keywords
    {
        Lexer lexer("ReLU LeakyReLU Sigmoid Tanh GELU SwiGLU Identity");
        auto toks = lexer.tokenize();
        assert(toks[0].type == TokenType::ACT_RELU);
        assert(toks[1].type == TokenType::ACT_LEAKY_RELU);
        assert(toks[4].type == TokenType::ACT_GELU);
        assert(toks[6].type == TokenType::ACT_IDENTITY);
    }

    // Optimizer keywords
    {
        Lexer lexer("AdamW Muon SGD");
        auto toks = lexer.tokenize();
        assert(toks[0].type == TokenType::OPT_ADAMW);
        assert(toks[1].type == TokenType::OPT_MUON);
        assert(toks[2].type == TokenType::OPT_SGD);
    }

    std::cout << "Lexer tests passed.\n";
    return 0;
}
