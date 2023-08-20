// Author: Zian Zhao
// For now, This pass is discarded
#include "heir/Passes/vec2memref/PromVecToMemref.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void FuncToHEIRPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithmeticDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}


class FHEExtractPattern final : public OpConversionPattern<FHEExtractOp>
{
public:
    using OpConversionPattern<FHEExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHEExtractOp op, typename FHEExtractOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        // if (auto st = dstType.dyn_cast_or_null<Float32Type>())
        // {
        //     // auto vector = op.getMemRef();
        //     int size = -28;
        //     auto tt = op.getVector().getType().dyn_cast<LWECipherVectorType>();
        //     if (tt.hasStaticShape() && tt.getShape().size() == 1)
        //         size = tt.getShape().front();
            
        //     auto batchedCipher = typeConverter->materializeTargetConversion(rewriter,
        //                                                                     op.getVector().getLoc(),
        //                                                                     LWECipherVectorType::get(getContext(),
        //                                                                                             st,
        //                                                                                             size),
        //                                                                     op.getVector());
        // }
            // SmallVector<Value, 8> indices(op.getMapOperands());
            // auto resultOperands =
            //     expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
            // if (!resultOperands)
            //     return failure();
        
            auto cOp = op.i().front().getDefiningOp<arith::ConstantOp>();
            if (!cOp)
            {
                emitError(op.getLoc(),
                        "tensor2fhe requires all tensor.extract indices used with tensors of fhe.secret to be constant!");
                return failure();
            }
            auto indexAttr = cOp.getValue().cast<IntegerAttr>();
            // llvm::outs() << indexAttr;
            // auto notBatchedCipher = typeConverter->materializeTargetConversion(rewriter,
            //                                                                     op.getMemRef().getLoc(),
            //                                                                     LWECipherVectorType::get(getContext(),
            //                                                                     op.getMemRef().getPlain
            //                                                                     ))
            // rewriter.replaceOpWithNewOp<FHEExtractOp>(op, op.getType(), vector, *resultOperands);
            // rewriter.replaceOpWithNewOp<FHEExtractOp>(op, dstType, op.vector(), op.i());
            // auto index = cOp.getValueAttr().cast<IntegerAttr>().getValue().getLimitedValue();
            // auto indexAttr = rewriter.getIndexAttr(index);
            rewriter.replaceOpWithNewOp<FHEExtractfinalOp>(op, dstType, op.vector(), indexAttr);
        // }
        return success();
    }
};

// class ArithConstantPattern final : public OpConversionPattern<arith::ConstantOp>
// {
// public:
//     using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

//     LogicalResult matchAndRewrite(
//         arith::ConstantOp op, typename arith::ConstantOpAdaptor adaptor, ConversionPattern &rewriter) const override
//     {
        
        
//         return success();
//     }
// };

// class FHEInsertPattern final : public OpConversionPattern<FHEInsertOp>
// {
// public:
//     using OpConversionPattern<FHEInsertOp>::OpConversionPattern;

//     LogicalResult matchAndRewrite(
//         FHEInsertOp op, typename FHEInsertOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
//     {
//         // auto indice = op.getIndices();

//         SmallVector<Value, 8> indices(op.getMapOperands());
//         auto maybeExpandedMap =
//             expandAffineMap(rewriter, op.getLoc(), op.getAffineMap(), indices);
//         if (!maybeExpandedMap)
//             return failure();

//         // Build memref.store valueToStore, memref[expandedMap.results].
//         llvm::outs() << op.getIndices().getBeginOperandIndex();
//         rewriter.replaceOpWithNewOp<FHEInsertOp>(
//             op, op.getValueToStore(), op.getMemRef(), *maybeExpandedMap);
        
//         return success();
//     }
// };

class FuncCallPattern final : public OpConversionPattern<func::CallOp>
{
protected:
    using OpConversionPattern<func::CallOp>::typeConverter;
public:
    using OpConversionPattern<func::CallOp>::OpConversionPattern;

    LogicalResult
    matchAndRewrite(func::CallOp op, typename func::CallOp::Adaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto resultType = getTypeConverter()->convertType(op.getResult(0).getType());
        if (!resultType)
            return failure();
        
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : adaptor.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                // assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }

            auto func_name = op.getCallee();

            rewriter.replaceOpWithNewOp<func::CallOp>(op, func_name, resultType, materialized_operands);
        }
        return success();
    }
};

class ReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
  using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override
  {
    llvm::outs() << "Enter ReturnPatttern \n";
    if (op.getNumOperands() != 1)
    {
      emitError(op.getLoc(), "Currently only single value return operations are supported.");
      return failure();
    }
    auto dstType = this->getTypeConverter()->convertType(op.getOperandTypes().front());
    if (!dstType)
      return failure();
    // if (auto bst = dstType.dyn_cast_or_null<Float32Type>())
    // {
    //   llvm::outs() << "Enter ReturnPatttern If\n";
      rewriter.setInsertionPoint(op);
      auto returnCipher = typeConverter->materializeTargetConversion(rewriter,
                                                                      op.getLoc(),
                                                                      dstType, op.operands());
                                                                    //   op.operands().front());
    //   rewriter.template replaceOpWithNewOp<func::ReturnOp>(op, returnCipher);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, returnCipher);

    // } // else do nothing
    return success();
  }
};

class EliminateVectorPattern final : public OpConversionPattern<AffineVectorLoadOp>
{
public: 
    using OpConversionPattern<AffineVectorLoadOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AffineVectorLoadOp op, typename AffineVectorLoadOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto old_memref = op.getMemRef();
        auto indices = op.getIndices();
        // auto map = op.getAffineMap();

        auto res = op.getResult();
        for (auto u : res.getUsers()) {
            if (auto loadOp = dyn_cast<AffineLoadOp>(u)) {
                auto new_memref = 
            }
        }
    }
}

class FunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // Compute the new signature of the function.
        TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
        SmallVector<Type> newResultTypes;
        if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return failure();
        if (typeConverter->convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
            return failure();
        auto new_functype = FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

        rewriter.startRootUpdate(op);
        op.setType(new_functype);
        for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
        {
            auto arg = *it;
            auto oldType = arg.getType();
            auto newType = typeConverter->convertType(oldType);
            arg.setType(newType);
            if (newType != oldType)
            {
                rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
            }
        }
        rewriter.finalizeRootUpdate(op);

        return success();
    }
};

// class EliminateConstant final : public OpConversionPattern<FHEExtractOp>
// {
// public:
//     using OpConversionPattern<FHEExtractOp>::OpConversionPattern;

//     LogicalResult matchAndRewrite(
//         FHEExtractOp op, typename FHEExtractOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
//     {
//         auto cOp = op.i().front().getDefiningOp<arith::ConstantOp>();
//         if (!cOp)
//         {
//             emitError(op.getLoc(),
//                     "tensor2fhe requires all tensor.extract indices used with tensors of fhe.secret to be constant!");
//             return failure();
//         }
//         // auto indexAttr = cOp.getValue().cast<IntegerAttr>();
//         auto indexAttr = cOp.getValue();
//         rewriter.replaceOpWithNewOp<FHEExtractOp>(op, op.getType(), op.vector(), indexAttr);
        
//         return success();
//     }
// };

void FuncToHEIRPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<Float32Type>())
            return llvm::Optional<Type>(LWECipherType::get(&getContext(), t));
        else if (t.isa<MemRefType>())
        {
            int size = -155;
            auto new_t = t.cast<MemRefType>();
            if (new_t.hasStaticShape() && new_t.getShape().size()==1) {
                size = new_t.getShape().front();
            }
            return llvm::Optional<Type>(LWECipherVectorType::get(&getContext(), new_t.getElementType(), size));
        }
        else
            return llvm::Optional<Type>(t);
    });
    type_converter.addTargetMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<Float32Type>())
            {
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None);
    });
    type_converter.addArgumentMaterialization([&] (OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<LWECipherType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<Float32Type>())
            {
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<LWECipherVectorType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<MemRefType>())
            {
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, ot, vs));
            }
        }
        return llvm::Optional<Value>(llvm::None);
    });
    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<Float32Type>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherType>())
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        else if (auto bst = t.dyn_cast_or_null<MemRefType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<LWECipherVectorType>())
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        return llvm::Optional<Value>(llvm::None);
    });
    
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithmeticDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalOp<FHEExtractOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::FuncOp>(op);
        for (auto t : op->getOperandTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : op->getResultTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getInputs())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getResults())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        return true;
    });
    target.addDynamicallyLegalOp<func::CallOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::CallOp>(op);
        for (auto t : op->getOperandTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : op->getResultTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getCalleeType().getInputs())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getCalleeType().getResults())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        return true;
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });

    target.addDynamicallyLegalOp<FHEExtractfinalOp>(
        [&](Operation *op) {return type_converter.isLegal(op->getResultTypes()); });

    IRRewriter rewriter(&getContext());
    
    // MemRefToHEIR
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
            FHEExtractPattern, FunctionConversionPattern, ReturnPattern, FuncCallPattern>(type_converter, patterns.getContext()); 
    
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    
}