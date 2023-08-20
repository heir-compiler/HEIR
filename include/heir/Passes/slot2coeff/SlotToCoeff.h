#ifndef HEIR_PASSES_SLOTTOCOEFF_H_
#define HEIR_PASSES_SLOTTOCOEFF_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "heir/IR/FHE/HEIRDialect.h"

struct SlotToCoeffPass : public mlir::PassWrapper<SlotToCoeffPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;
    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "slot2coeff";
    }
};

#endif // HEIR_PASSES_SLOTTOCOEFF_H_
