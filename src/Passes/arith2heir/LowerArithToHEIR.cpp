// Author: Zian Zhao
#include "heir/Passes/arith2heir/LowerArithToHEIR.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void LowerArithToHEIRPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithmeticDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<memref::MemRefDialect>();
}

// Transform arith::AddFOp/MulFOp/SubFOp into corresponding homomorphic
// operators: heir::LWEAddOp/LWEMulOp/LWESubOp. Convert the data type of
// input/output of the operators
template <typename OpType>
class ArithBasicPattern final : public OpConversionPattern<OpType>
{
protected:
    using OpConversionPattern<OpType>::typeConverter;

public:
    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpType op, typename OpType::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        // Multiplications
        if (std::is_same<OpType, MulFOp>())
        {
            rewriter.replaceOpWithNewOp<LWEMulOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }

        // Additions
        else if (std::is_same<OpType, AddFOp>())
        {
            rewriter.replaceOpWithNewOp<LWEAddOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        
        // Substractions
        else if (std::is_same<OpType, SubFOp>())
        {
            rewriter.replaceOpWithNewOp<LWESubOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        return failure();
    };
};

// Transform vector allocation Op into corresponding homomorphic
// operators: heir::FHEDefineOp. Convert the output data type of the operators
class ArithAllocatePattern final : public OpConversionPattern<memref::AllocaOp>
{
public:
    using OpConversionPattern<memref::AllocaOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        memref::AllocaOp op, typename memref::AllocaOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        
        rewriter.replaceOpWithNewOp<FHEDefineOp>(op, dstType);

        return success();
    }
};

// Transform arith::CmpFOp into corresponding homomorphic
// operators: heir::FHECmpOp. Convert the data type of
// input/output of the operators
class ArithComparePattern final : public OpConversionPattern<CmpFOp>
{
public:
    using OpConversionPattern<CmpFOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        CmpFOp op, typename CmpFOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType) 
            return failure();
        
        // Materialize the operands where necessary
        auto lhs = op.getLhs();
        auto rhs = op.getRhs();
        Value new_lhs, new_rhs;

        // Convert the type of operands(inputs)
        auto operandDstType = typeConverter->convertType(lhs.getType());
        if (!operandDstType)
                return failure();
        if (lhs.getType() != operandDstType)
        {
            new_lhs = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, lhs);
            assert(new_lhs && "Type Conversion must not fail");
        }
        else
        {
            new_lhs = lhs;
        }
        operandDstType = typeConverter->convertType(rhs.getType());
        if (rhs.getType() != operandDstType)
        {
            new_rhs = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, rhs);
            assert(new_rhs && "Type Conversion must not fail");
        }
        else
        {
            new_rhs = rhs;
        }

        arith::CmpFPredicate predicate = op.getPredicate();

        rewriter.replaceOpWithNewOp<FHECmpOp>(op, TypeRange(dstType), predicate, 
                                                new_lhs, new_rhs);
        return success();
    }
    
};

// Transform arith::SelectOp into corresponding homomorphic
// operators: heir::FHESelectOp. Convert the data type of
// input/output of the operators
class ArithSelectPattern final : public OpConversionPattern<SelectOp>
{
protected:
    using OpConversionPattern<SelectOp>::typeConverter;

public:
    using OpConversionPattern<SelectOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        SelectOp op, typename SelectOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // Materialize the operands where necessary
        // Value materialized_true, materialized_false, materialized_condition;
        Value trueValue = op.getTrueValue();
        Value falseValue = op.getFalseValue();
        Value condition = op.getCondition();

        auto trueDstType = typeConverter->convertType(trueValue.getType());
        auto falseDstType = typeConverter->convertType(falseValue.getType());
        auto conDstType = typeConverter->convertType(condition.getType());

        if (!trueDstType || !falseDstType || !conDstType)
            return failure();

        auto materialized_true = typeConverter->materializeTargetConversion(rewriter, op.getLoc(),
                                                trueDstType, trueValue);
        auto materialized_false = typeConverter->materializeTargetConversion(rewriter, op.getLoc(),
                                                falseDstType, falseValue);
        auto materialized_condition = typeConverter->materializeTargetConversion(rewriter, op.getLoc(),
                                                conDstType, condition);

        
        rewriter.replaceOpWithNewOp<FHESelectOp>(op, dstType, materialized_condition, 
                                    materialized_true, materialized_false);
        return success();
    };
};

// Transform arith::Constantp into corresponding homomorphic
// operators: heir::MaterializeOp for encoding. Convert the data type of
// input/output of the operators
LogicalResult ArithConstOperation(IRRewriter &rewriter, MLIRContext *context, ConstantOp op, TypeConverter typeConverter)
{
    if (op.getType() != Float32Type::get(rewriter.getContext()))
        return success();
    
    for (OpOperand &use : op->getUses()) {
        Operation *userOp = use.getOwner();
        if (FHEMaterializeOp mat_op = dyn_cast<FHEMaterializeOp>(userOp)) {
            if (!mat_op.use_empty()) {
                auto message = op.getValueAttr().dyn_cast_or_null<FloatAttr>();
                FloatAttr new_message = message;
                rewriter.setInsertionPoint(mat_op);
                rewriter.replaceOpWithNewOp<FHEEncodeOp>(mat_op, PlainType::get(rewriter.getContext()), new_message);
            }
        }
    }

    return success();
}

// Add FHEMaterializeOp for FHELUTOp in case data type inconsistant during type conversion
LogicalResult FHELUTTypeOperation(IRRewriter &rewriter, MLIRContext *context, FHELUTOp op, TypeConverter typeConverter)
{
    auto inputType = op.input().getType();
    auto outputType = op.getType();
    auto socType = typeConverter.convertType(inputType);
    auto dstType = typeConverter.convertType(outputType);

    rewriter.setInsertionPoint(op);
    auto forward_mat_op = rewriter.create<FHEMaterializeOp>(op.getLoc(), socType, op.input());
    auto newLUTOp = rewriter.create<FHELUTOp>(forward_mat_op.getLoc(), dstType, forward_mat_op.result());
    rewriter.replaceOpWithNewOp<FHEMaterializeOp>(op, outputType, newLUTOp.result());
    
    return success();
}

// Convert a set of arith Dialect operators to heir operators
void LowerArithToHEIRPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    // Add type converter to convert plaintext data type to LWECiphertext type
    type_converter.addConversion([&](Type t) {
        if (t.isa<Float32Type>())
            return llvm::Optional<Type>(LWECipherType::get(&getContext(), t));
        else if (t.isa<IntegerType>())
            return llvm::Optional<Type>(LWECipherType::get(&getContext(), Float32Type::getF32(&getContext())));
        else if (t.isa<MemRefType>())
        {
            int size = -155;
            auto new_t = t.cast<MemRefType>();
            if (new_t.hasStaticShape() && new_t.getShape().size()==1) {
                size = new_t.getShape().front();
                return llvm::Optional<Type>(LWECipherVectorType::get(&getContext(), new_t.getElementType(), size));
            }
            else if (new_t.hasStaticShape() && new_t.getShape().size()==2) {
                int row = new_t.getShape().front();
                int column = new_t.getShape().back();
                return llvm::Optional<Type>(LWECipherMatrixType::get(&getContext(), new_t.getElementType(), row, column));
            }
            else 
                return llvm::Optional<Type>(t);
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
            else if (old_type.dyn_cast_or_null<IntegerType>())
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
        else if (auto ot = t.dyn_cast_or_null<LWECipherMatrixType>())
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
            else if (old_type.dyn_cast_or_null<IntegerType>())
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
        else if (auto ot = t.dyn_cast_or_null<LWECipherMatrixType>())
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
        else if (auto bst = t.dyn_cast_or_null<IntegerType>())
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
            else if (auto ot = old_type.dyn_cast_or_null<LWECipherMatrixType>())
                return llvm::Optional<Value>(builder.create<FHEMaterializeOp>(loc, bst, vs));
        }
        return llvm::Optional<Value>(llvm::None);
    });
    
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, scf::SCFDialect, ArithmeticDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();

    IRRewriter rewriter(&getContext());
    
    // ArithToHEIR
    target.addIllegalOp<MulFOp>();
    target.addIllegalOp<AddFOp>();
    target.addIllegalOp<SubFOp>();
    target.addIllegalOp<CmpFOp>();
    target.addIllegalOp<memref::AllocaOp>();

    mlir::RewritePatternSet selectPatterns(&getContext());
    ConversionTarget newtarget(getContext());
    newtarget.addLegalDialect<AffineDialect, func::FuncDialect, scf::SCFDialect, ArithmeticDialect>();
    newtarget.addLegalDialect<memref::MemRefDialect>();
    newtarget.addLegalDialect<HEIRDialect>();
    newtarget.addLegalOp<ModuleOp>();
    newtarget.addIllegalOp<SelectOp>();
    selectPatterns.add<ArithSelectPattern>(type_converter, selectPatterns.getContext()); 
    
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), newtarget, std::move(selectPatterns))))
        signalPassFailure();
    
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
            ArithBasicPattern<MulFOp>, ArithBasicPattern<AddFOp>, ArithBasicPattern<SubFOp>,
            ArithAllocatePattern, ArithComparePattern>(type_converter, patterns.getContext());
    
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
    
    // The third Pattern: Constant--Encode Pattern to convert ConstantOp
    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if(f.walk([&](Operation *op)
                {
        if (ConstantOp const_op = llvm::dyn_cast_or_null<ConstantOp>(op)) {
            if (ArithConstOperation(rewriter, &getContext(), const_op, type_converter).failed()) 
                return WalkResult::interrupt();
        }
        else if (FHELUTOp lut_op = llvm::dyn_cast_or_null<FHELUTOp>(op)) {
            if (FHELUTTypeOperation(rewriter, &getContext(), lut_op, type_converter).failed())
                return WalkResult::interrupt();
        }
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
    }

}