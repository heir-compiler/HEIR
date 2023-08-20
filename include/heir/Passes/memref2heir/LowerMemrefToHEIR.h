#ifndef HEIR_PASSES_MEMREF2HEIR_LOWERMEMREFTOHEIR_H_
#define HEIR_PASSES_MEMREF2HEIR_LOWERMEMREFTOHEIR_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct LowerMemrefToHEIRPass : public mlir::PassWrapper<LowerMemrefToHEIRPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "memref2heir";
    }
};

#endif // HEIR_PASSES_MEMREF2HEIR_LOWERMEMREFTOHEIR_H_
