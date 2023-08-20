#ifndef HEIR_PASSES_VEC2MEMREF_PROMVECTOMEMREF_H_
#define HEIR_PASSES_VEC2MEMREF_PROMVECTOMEMREF_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct PromVecToMemrefPass : public mlir::PassWrapper<PromVecToMemrefPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "vec2memref";
    }
};

#endif // HEIR_PASSES_VEC2MEMREF_PROMVECTOMEMREF_H_
