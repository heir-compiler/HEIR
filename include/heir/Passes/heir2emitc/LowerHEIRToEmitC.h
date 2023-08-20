#ifndef HEIR_PASSES_HEIR2EMITC_LOWERHEIRTOEMITC_H_
#define HEIR_PASSES_HEIR2EMITC_LOWERHEIRTOEMITC_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct LowerHEIRToEmitCPass : public mlir::PassWrapper<LowerHEIRToEmitCPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "heir2emitc";
    }
};

#endif // HEIR_PASSES_HEIR2EMITC_LOWERHEIRTOEMITC_H_
