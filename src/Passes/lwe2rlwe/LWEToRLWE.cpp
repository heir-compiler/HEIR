// Author: Zian Zhao
#include <queue>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "heir/Passes/lwe2rlwe/LWEToRLWE.h"

using namespace mlir;
using namespace heir;

void LWEToRLWEPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<heir::HEIRDialect,
                  mlir::AffineDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect>();
}

// Transform the batched LWE addition operator into RLWE addition
LogicalResult LWEAddToRLWEOperation(IRRewriter &rewriter, MLIRContext *context, LWEAddOp op, TypeConverter typeConverter)
{
    rewriter.setInsertionPoint(op);

    auto dstType = typeConverter.convertType(op.getType());
    if (!dstType)
        return failure();
    
    llvm::SmallVector<Value> materialized_operands;
    for (Value o : op.getOperands())
    {
        auto operandDstType = typeConverter.convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType) {
            auto new_operand = typeConverter.materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else {
            materialized_operands.push_back(o);
        }      
    }

    rewriter.replaceOpWithNewOp<LWEAddOp>(op, dstType, materialized_operands);
    return success();
}

// Transform the batched LWE substraction operator into RLWE substraction
LogicalResult LWESubToRLWEOperation(IRRewriter &rewriter, MLIRContext *context, LWESubOp op, TypeConverter typeConverter)
{
    rewriter.setInsertionPoint(op);

    auto dstType = typeConverter.convertType(op.getType());
    if (!dstType)
        return failure();
    
    llvm::SmallVector<Value> materialized_operands;
    for (Value o : op.getOperands())
    {
        auto operandDstType = typeConverter.convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType) {
            auto new_operand = typeConverter.materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else {
            materialized_operands.push_back(o);
        }
    }
    
    rewriter.replaceOpWithNewOp<LWESubOp>(op, dstType, materialized_operands);
    return success();
    
}

// Transform the batched LWE multiplication operator into RLWE multiplication
LogicalResult LWEMulToRLWEOperation(IRRewriter &rewriter, MLIRContext *context, LWEMulOp op, TypeConverter typeConverter)
{
    rewriter.setInsertionPoint(op);

    auto dstType = typeConverter.convertType(op.getType());
    if (!dstType)
        return failure();
    
    llvm::SmallVector<Value> materialized_operands;
    for (Value o : op.getOperands())
    {
        auto operandDstType = typeConverter.convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType) {
            auto new_operand = typeConverter.materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else {
            materialized_operands.push_back(o);
        }
    }
    
    rewriter.replaceOpWithNewOp<LWEMulOp>(op, dstType, materialized_operands);
    return success();
    
}

// Tranform a program computed in pure LWE ciphertexts into RLWE ciphertexts 
// after batching optimizations
void LWEToRLWEPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<LWECipherVectorType>())
        {
            int size = -155;
            auto new_t = t.cast<LWECipherVectorType>();
            size = new_t.getSize();
            return llvm::Optional<Type>(RLWECipherType::get(&getContext(), new_t.getPlaintextType(), size));
        }
        else
            return llvm::Optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<RLWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<LWECipherVectorType>())
            {
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None);
    });
    type_converter.addArgumentMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<RLWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<LWECipherVectorType>())
            {
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None);
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<RLWECipherType>())
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        return llvm::Optional<Value>(llvm::None);
    });

    
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        if (f.walk([&](Operation *op)
                {
        if (LWESubOp sub_op = llvm::dyn_cast_or_null<LWESubOp>(op)) {
            if (LWESubToRLWEOperation(rewriter, &getContext(), sub_op, type_converter).failed())
            return WalkResult::interrupt();
        } 
        else if (LWEAddOp add_op = llvm::dyn_cast_or_null<LWEAddOp>(op)) {
            if (LWEAddToRLWEOperation(rewriter, &getContext(), add_op, type_converter).failed())
            return WalkResult::interrupt();
        } 
        else if (LWEMulOp mul_op = llvm::dyn_cast_or_null<LWEMulOp>(op)) {
            if (LWEMulToRLWEOperation(rewriter, &getContext(), mul_op, type_converter).failed())
            return WalkResult::interrupt();
        } 
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();

        // int argNum = f.getNumArguments();
        // for (int i = 0; i < argNum; i++) {
        //     Value arg = f.getArgument(i);
        //     auto dstType = type_converter.convertType(arg.getType());
        //     arg.setType(dstType);
        //     // Value new_arg = type_converter.materializeTargetConversion(rewriter, f.getLoc(), dstType, arg);
        //     // arg.replaceAllUsesWith(new_arg);
        // }
        // Convert types of the function arguments
        func::FuncOp op = f;
        TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
        SmallVector<Type> newResultTypes;
        if (failed(type_converter.convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            signalPassFailure();
        if (type_converter.convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
            signalPassFailure();
        auto new_functype = FunctionType::get(&getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

        rewriter.startRootUpdate(op);
        op.setType(new_functype);
        for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
        {
            auto arg = *it;
            auto oldType = arg.getType();
            auto newType = type_converter.convertType(oldType);
            arg.setType(newType);
            if (newType != oldType)
            {
                rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                auto m_op = type_converter.materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
            }
        }
        rewriter.finalizeRootUpdate(op);
    }

}