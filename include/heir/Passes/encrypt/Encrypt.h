#ifndef HEIR_ENCRYPT_TD
#define HEIR_ENCRYPT_TD

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct EncryptPass : public mlir::PassWrapper<EncryptPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "encrypt";
    }
};

#endif // HEIR_ENCRYPT_TD
