#ifndef HEIR_PASSES_GENPARM_H_
#define HEIR_PASSES_GENPARM_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct GenerateParmPass : public mlir::PassWrapper<GenerateParmPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "genparm";
    }
};

#endif // HEIR_PASSES_GENPARM_H_
