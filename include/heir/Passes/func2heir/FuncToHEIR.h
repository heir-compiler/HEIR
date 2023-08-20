#ifndef HEIR_PASSES_FUNC2HEIR_LOWERFUNCTOHEIR_H_
#define HEIR_PASSES_FUNC2HEIR_LOWERFUNCTOHEIR_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct FuncToHEIRPass : public mlir::PassWrapper<FuncToHEIRPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "func2heir";
    }
};

#endif // HEIR_PASSES_FUNC2HEIR_LOWERFUNCTOHEIR_H_
