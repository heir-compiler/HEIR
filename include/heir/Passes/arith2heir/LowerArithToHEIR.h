#ifndef HEIR_PASSES_ARITH2HEIR_LOWERARITHTOHEIR_H_
#define HEIR_PASSES_ARITH2HEIR_LOWERARITHTOHEIR_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct LowerArithToHEIRPass : public mlir::PassWrapper<LowerArithToHEIRPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "arith2heir";
    }
};

#endif // HEIR_PASSES_ARITH2HEIR_LOWERARITHTOHEIR_H_
