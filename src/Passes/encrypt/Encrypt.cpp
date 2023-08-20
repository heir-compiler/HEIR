// Author: Zian Zhao
#include "heir/Passes/func2heir/FuncToHEIR.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "heir/Passes/encrypt/Encrypt.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void EncryptPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithmeticDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}

// Based on the pattern in arith2heir pass, convert the MaterializeOp into
// FHEEncryptOp to change a plaintext to ciphertext
// TODO: Actually, we just need to encode it rather than encrypt it, but we face the
// limitation of the undelying FHE library.
class FHEEncryptConversionPattern final : public OpConversionPattern<FHEMaterializeOp>
{
public:
    using OpConversionPattern<FHEMaterializeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEMaterializeOp op, typename FHEMaterializeOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        Value operand = op.getOperand();
        if (operand.getType().dyn_cast_or_null<Float32Type>()) {
            rewriter.replaceOpWithNewOp<FHEEncryptOp>(op, op.getType(), operand);
        }

        return success();
    }
};

void EncryptPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithmeticDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<FHEMaterializeOp>();
    

    IRRewriter rewriter(&getContext());
    
    // MemRefToHEIR
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FHEEncryptConversionPattern>(patterns.getContext()); 
    
    if (mlir::failed(mlir::applyFullConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    
}