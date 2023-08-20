#ifndef HEIR_PASSES_NARY_H_
#define HEIR_PASSES_NARY_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct NaryPass : public mlir::PassWrapper<NaryPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;
    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "nary";
    }
};

#endif // HEIR_PASSES_NARY_H_
