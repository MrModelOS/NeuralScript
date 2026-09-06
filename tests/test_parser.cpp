#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include <cassert>
#include <iostream>

using namespace ns;

int main() {
    // Network declaration with pipeline
    {
        std::string src = R"(
type Batch = Dynamic
type Features = 784
type Classes = 10

network DenseBlock {
    input:  Tensor[B, InDim] float32
    output: Tensor[B, OutDim] float32

    layer fc1 = Dense(in: InDim, out: 512, activation: ReLU)
    layer drop = Dropout(rate: 0.1)
    layer fc2 = Dense(in: 512, out: OutDim, activation: Identity)

    forward(x) {
        return x -> fc1 -> drop -> fc2
    }
}
)";
        Lexer lexer(src);
        auto toks = lexer.tokenize();
        Parser parser(toks);
        Program prog = parser.parse_program();

        assert(prog.top_level.size() >= 1);
        bool found_network = false;
        for (auto& stmt : prog.top_level) {
            if (stmt->kind == Stmt::NETWORK_DECL) {
                found_network = true;
                assert(stmt->network_name == "DenseBlock");
                assert(stmt->layers.size() == 3);
                assert(stmt->layers[0]->layer_name == "fc1");
                assert(stmt->layers[0]->layer_type == "Dense");
            }
        }
        assert(found_network);
    }

    // Function with grad block and matmul
    {
        std::string src = R"(
fn train_step(model, batch_x, batch_y, opt) {
    grad(model) {
        var preds = model.forward(batch_x)
        var loss = cross_entropy(preds, batch_y)
    }
    opt.step(model)
}
)";
        Lexer lexer(src);
        auto toks = lexer.tokenize();
        Parser parser(toks);
        Program prog = parser.parse_program();
        assert(prog.top_level.size() == 1);
        assert(prog.top_level[0]->kind == Stmt::FN_DECL);
        assert(prog.top_level[0]->fn_name == "train_step");
    }

    // Type declarations
    {
        std::string src = "type Features = 784\ntype Classes = 10\n";
        Lexer lexer(src);
        auto toks = lexer.tokenize();
        Parser parser(toks);
        Program prog = parser.parse_program();
        assert(prog.top_level.size() == 2);
        assert(prog.top_level[0]->kind == Stmt::TYPE_DECL);
        assert(prog.top_level[0]->alias_name == "Features");
    }

    std::cout << "Parser tests passed.\n";
    return 0;
}
