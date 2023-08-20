#ifndef HEIR_PASSES_ARITH2HEIR_UNROLLLOOP_H_
#define HEIR_PASSES_ARITH2HEIR_UNROLLLOOP_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct UnrollLoopPass : public mlir::PassWrapper<UnrollLoopPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "loop-unroll";
    }
};

#endif // HEIR_PASSES_ARITH2HEIR_UNROLLLOOP_H_
