#include "ns/parser/ast.hpp"

namespace ns {

std::string dtype_to_string(Dtype dt) {
    switch (dt) {
        case Dtype::Float16: return "float16";
        case Dtype::Float32: return "float32";
        case Dtype::Float64: return "float64";
        case Dtype::Int8: return "int8";
        case Dtype::Int16: return "int16";
        case Dtype::Int32: return "int32";
        case Dtype::Int64: return "int64";
        case Dtype::FP8: return "fp8";
        case Dtype::FP4: return "fp4";
        case Dtype::Bool: return "bool";
    }
    return "unknown";
}

Dtype token_to_dtype(TokenType tt) {
    switch (tt) {
        case TokenType::DTYPE_FLOAT16: return Dtype::Float16;
        case TokenType::DTYPE_FLOAT32: return Dtype::Float32;
        case TokenType::DTYPE_FLOAT64: return Dtype::Float64;
        case TokenType::DTYPE_INT8: return Dtype::Int8;
        case TokenType::DTYPE_INT16: return Dtype::Int16;
        case TokenType::DTYPE_INT32: return Dtype::Int32;
        case TokenType::DTYPE_INT64: return Dtype::Int64;
        case TokenType::DTYPE_FP8: return Dtype::FP8;
        case TokenType::DTYPE_FP4: return Dtype::FP4;
        case TokenType::DTYPE_BOOL: return Dtype::Bool;
        default: return Dtype::Float32;
    }
}

std::string dim_expr_to_string(const DimExpr& dim) {
    switch (dim.kind) {
        case DimExpr::CONST: return std::to_string(dim.const_value);
        case DimExpr::SYMBOLIC: return dim.symbolic_name;
        case DimExpr::DYNAMIC: return "Dynamic";
    }
    return "?";
}

std::string tensor_type_to_string(const TensorType& tt) {
    std::string result = "Tensor[";
    for (size_t i = 0; i < tt.dims.size(); i++) {
        if (i > 0) result += ", ";
        result += dim_expr_to_string(tt.dims[i]);
    }
    result += "] " + dtype_to_string(tt.dtype);
    return result;
}

} // namespace ns
