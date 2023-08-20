#ifndef HEIR_PASSES_BRANCHUNFOLD_H_
#define HEIR_PASSES_BRANCHUNFOLD_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct BranchUnfoldPass : public mlir::PassWrapper<BranchUnfoldPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "unfold";
    }
};

#endif // HEIR_PASSES_BRANCHUNFOLD_H_
