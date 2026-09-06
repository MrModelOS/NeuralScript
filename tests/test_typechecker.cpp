#include "ns/lexer/lexer.hpp"
#include "ns/parser/parser.hpp"
#include "ns/typechecker/shape_checker.hpp"
#include <cassert>
#include <iostream>

using namespace ns;

static bool check_program(const std::string& src, std::vector<ShapeError>& errs) {
    Lexer lexer(src);
    auto toks = lexer.tokenize();
    Parser parser(toks);
    Program prog = parser.parse_program();
    ShapeChecker checker;
    bool ok = checker.check(prog);
    errs = checker.errors();
    return ok;
}

int main() {
    // Valid matmul: input [B, 784] @ [784, 10] -> [B, 10]
    {
        std::string src = R"(
var input: Tensor[B, 784] float32
var weights: Tensor[784, 10] float32
var logits = input @ weights
)";
        std::vector<ShapeError> errs;
        bool ok = check_program(src, errs);
        assert(ok);
    }

    // Invalid matmul: mismatched inner dimensions should be rejected
    {
        std::string src = R"(
var a: Tensor[4, 3] float32
var b: Tensor[5, 6] float32
var c = a @ b
)";
        std::vector<ShapeError> errs;
        bool ok = check_program(src, errs);
        assert(!ok);
        bool found_mismatch = false;
        for (auto& e : errs) {
            if (e.message.find("mismatch") != std::string::npos) found_mismatch = true;
        }
        assert(found_mismatch);
    }

    // Dynamic batch compiles: dims unify
    {
        std::string src = R"(
type Batch = Dynamic
var x: Tensor[Batch, 784] float32
var w: Tensor[784, 10] float32
var y = x @ w
)";
        std::vector<ShapeError> errs;
        bool ok = check_program(src, errs);
        assert(ok);
    }

    // cross_entropy shape agreement
    {
        std::string src = R"(
var preds: Tensor[B, 10] float32
var labels: Tensor[B, 10] float32
var loss = cross_entropy(preds, labels)
)";
        std::vector<ShapeError> errs;
        bool ok = check_program(src, errs);
        assert(ok);
    }

    // ---- Layer chain shape inference (Phase 1) ----

    // Correct: fc(out=Classes=10) matches declared output [B, 10]
    {
        std::string src = R"(
type Batch = Dynamic
type Features = 784
type Classes = 10

network Net {
    input:  Tensor[Batch, Features] float32
    output: Tensor[Batch, Classes] float32
    layer fc = Dense(in: Features, out: Classes, activation: ReLU)
    forward(x) {
        return x -> fc
    }
}
)";
        std::vector<ShapeError> errs;
        assert(check_program(src, errs));
    }

    // Error: layer output [B, 512] mismatches declared output [B, 10]
    {
        std::string src = R"(
type Batch = Dynamic
type Features = 784
type Classes = 10

network Net {
    input:  Tensor[Batch, Features] float32
    output: Tensor[Batch, Classes] float32
    layer fc = Dense(in: Features, out: 512, activation: ReLU)
    forward(x) {
        return x -> fc
    }
}
)";
        std::vector<ShapeError> errs;
        bool ok = check_program(src, errs);
        assert(!ok);
        bool found = false;
        for (auto& e : errs) {
            if (e.message.find("10 vs 512") != std::string::npos ||
                e.message.find("512") != std::string::npos) found = true;
        }
        assert(found);
    }

    // Shape-preserving layers: Dropout + LayerNorm keep [B, 512]
    {
        std::string src = R"(
type Batch = Dynamic
type H = 512
type Classes = 10

network Net {
    input:  Tensor[Batch, H] float32
    output: Tensor[Batch, Classes] float32
    layer ln = LayerNorm()
    layer drop = Dropout(rate: 0.1)
    layer fc = Dense(in: H, out: Classes, activation: Identity)
    forward(x) {
        return x -> ln -> drop -> fc
    }
}
)";
        std::vector<ShapeError> errs;
        assert(check_program(src, errs));
    }

    // Embedding expands [B, seq] -> [B, seq, d_model]; verify against output
    {
        std::string src = R"(
type Batch = Dynamic
type Seq = 128
type DModel = 64

network Emb {
    input:  Tensor[Batch, Seq] int64
    output: Tensor[Batch, Seq, DModel] float32
    layer tok_emb = Embedding(vocab_size: 10000, d_model: DModel)
    forward(ids) {
        return ids -> tok_emb
    }
}
)";
        std::vector<ShapeError> errs;
        assert(check_program(src, errs));
    }

    std::cout << "Type checker tests passed.\n";
    return 0;
}
