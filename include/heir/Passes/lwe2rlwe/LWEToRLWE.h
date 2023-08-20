#ifndef HEIR_PASSES_LWETORLWE_H_
#define HEIR_PASSES_LWETORLWE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct LWEToRLWEPass : public mlir::PassWrapper<LWEToRLWEPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "lwe2rlwe";
    }
};

#endif // HEIR_PASSES_LWETORLWE_H_
