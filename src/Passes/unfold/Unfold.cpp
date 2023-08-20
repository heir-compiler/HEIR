// Author: Zian Zhao
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "heir/Passes/unfold/BranchUnfold.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void BranchUnfoldPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithmeticDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<scf::SCFDialect>();
    registry.insert<HEIRDialect>();
}

// Transform CmpFOp into func::CallOp to call TFHE LUT, abandoned
class ArithCmpFPattern final : public OpConversionPattern<CmpFOp>
{
public:
    using OpConversionPattern<CmpFOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        CmpFOp op, typename CmpFOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        CmpFPredicate predicate = op.getPredicate();
        Value firstOperand = op.getOperand(0);
        Value secondOperand = op.getOperand(1);
        auto subOp = rewriter.create<SubFOp>(op.getLoc(), firstOperand, secondOperand);
        if (predicate == CmpFPredicate::OLT) {
            auto lutOp = rewriter.create<func::CallOp>(op.getLoc(), subOp.getType(), StringRef("lut_ltz"), 
                         ValueRange(subOp.getResult()));
            auto castOp = rewriter.create<FPToUIOp>(op.getLoc(), op.getType(), lutOp.getResult(0));
            rewriter.replaceOp(op, castOp.getResult());
        }
        else if (predicate == CmpFPredicate::OGT) {
            auto lutOp = rewriter.create<func::CallOp>(op.getLoc(), subOp.getType(), StringRef("lut_gtz"), 
                         ValueRange(subOp.getResult()));
            auto castOp = rewriter.create<FPToUIOp>(op.getLoc(), op.getType(), lutOp.getResult(0));
            rewriter.replaceOp(op, castOp.getResult());
        }
        else if (predicate == CmpFPredicate::OLE) {

        }
        else if (predicate == CmpFPredicate::OGE) {

        }
        return success();
    }
};

// Sometimes SelectOp exists together with CmpFOp, we need to 
// eliminate it 
class FHESelectPattern final : public OpConversionPattern<FHESelectOp>
{
public:
    using OpConversionPattern<FHESelectOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        FHESelectOp op, typename FHESelectOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        Value cond = op.condition();
        Value trueValue = op.true_value();
        Value falseValue = op.false_value();

        if (auto c_op = cond.getDefiningOp<FHECmpOp>()) {

            rewriter.setInsertionPoint(op);
            Value cmp_lhs = c_op.lhs();
            Value cmp_rhs = c_op.rhs();
            llvm::SmallVector<Value> c_operands;
            c_operands.push_back(cmp_lhs);
            c_operands.push_back(cmp_rhs);
            auto predicate = c_op.predicate();

            // For calculating min_value or max_value
            if (trueValue == cmp_rhs && falseValue == cmp_lhs) {
                // llvm::outs() << "can you see me\n";
                auto add_op = rewriter.create<LWEAddOp>(op.getLoc(), c_op.getType(), c_operands);
                auto sub_op = rewriter.create<LWESubOp>(add_op.getLoc(), c_op.getType(), c_operands);
                auto add_lut_op = rewriter.create<FHELUTForAddOp>(sub_op.getLoc(), add_op.getType(), add_op.output());
                auto sub_lut_op = rewriter.create<FHELUTForSubOp>(add_lut_op.getLoc(), sub_op.getType(), sub_op.output());
                llvm::SmallVector<Value> lut_operands;
                lut_operands.push_back(add_lut_op.result());
                lut_operands.push_back(sub_lut_op.result());

                // predicate = ogt
                if (predicate == CmpFPredicate::OGT) {
                    auto finalSub_op = rewriter.create<LWESubOp>(sub_lut_op.getLoc(), sub_lut_op.getType(), lut_operands);
                    op->replaceAllUsesWith(finalSub_op);
                    rewriter.eraseOp(op);
                    rewriter.eraseOp(c_op);
                }
                // predicate = olt
                else if (predicate == CmpFPredicate::OLT) {
                    auto finalAdd_op = rewriter.create<LWEAddOp>(sub_lut_op.getLoc(), sub_lut_op.getType(), lut_operands);
                    op->replaceAllUsesWith(finalAdd_op);
                    rewriter.eraseOp(op);
                    rewriter.eraseOp(c_op);
                } 
            }
            // For general cases, but may cause level problem
            else {
                auto cmpSub_op = rewriter.create<LWESubOp>(c_op.getLoc(), c_op.getType(), c_operands);
                // Predicate = ogt
                if (predicate == CmpFPredicate::OGT) {
                    auto cmpLUT_op = rewriter.create<FHELUTForGTOp>(cmpSub_op.getLoc(), cmpSub_op.getType(), cmpSub_op.getResult());
                    llvm::SmallVector<Value> trueMul_operands;
                    trueMul_operands.push_back(cmpLUT_op.result());
                    trueMul_operands.push_back(trueValue);
                    auto trueMul_op = rewriter.create<LWEMulOp>(cmpLUT_op.getLoc(), cmpLUT_op.getType(), trueMul_operands);
                    // FloatAttr oneAttr;
                    // oneAttr.get(Float32Type::get(rewriter.getContext()), 1.);
                    APFloat f1(1.0f);
                    auto encode_op = rewriter.create<FHEEncodeOp>(trueMul_op.getLoc(), 
                                                        PlainType::get(rewriter.getContext()), f1);
                    llvm::SmallVector<Value> sub_operands;
                    sub_operands.push_back(encode_op.getResult());
                    sub_operands.push_back(cmpLUT_op.result());
                    auto sub_op = rewriter.create<LWESubOp>(encode_op.getLoc(), cmpLUT_op.getType(), sub_operands);
                    llvm::SmallVector<Value> falseMul_operands;
                    falseMul_operands.push_back(sub_op.getResult());
                    falseMul_operands.push_back(falseValue);
                    auto falseMul_op = rewriter.create<LWEMulOp>(sub_op.getLoc(), sub_op.getType(), falseMul_operands);
                    llvm::SmallVector<Value> finalAdd_operands;
                    finalAdd_operands.push_back(trueMul_op.getResult());
                    finalAdd_operands.push_back(falseMul_op.getResult());
                    auto finalAdd_op = rewriter.create<LWEAddOp>(falseMul_op.getLoc(), falseMul_op.getType(), finalAdd_operands);
                    auto finalLUT_op = rewriter.create<FHELUTOp>(finalAdd_op.getLoc(), finalAdd_op.getType(), finalAdd_op.getResult());

                    op->replaceAllUsesWith(finalLUT_op);
                    rewriter.eraseOp(op);
                    c_op->replaceAllUsesWith(cmpLUT_op);
                    rewriter.eraseOp(c_op);

                } else if (predicate == CmpFPredicate::OLT) {
                    auto cmpLUT_op = rewriter.create<FHELUTForLTOp>(cmpSub_op.getLoc(), cmpSub_op.getType(), cmpSub_op.getResult());
                    llvm::SmallVector<Value> trueMul_operands;
                    trueMul_operands.push_back(cmpLUT_op.result());
                    trueMul_operands.push_back(trueValue);
                    auto trueMul_op = rewriter.create<LWEMulOp>(cmpLUT_op.getLoc(), cmpLUT_op.getType(), trueMul_operands);
                    // FloatAttr oneAttr;
                    // oneAttr.get(Float32Type::get(rewriter.getContext()), 1.);
                    APFloat f1(1.0f);
                    auto encode_op = rewriter.create<FHEEncodeOp>(trueMul_op.getLoc(), 
                                                        PlainType::get(rewriter.getContext()), f1);
                    llvm::SmallVector<Value> sub_operands;
                    sub_operands.push_back(encode_op.getResult());
                    sub_operands.push_back(cmpLUT_op.result());
                    auto sub_op = rewriter.create<LWESubOp>(encode_op.getLoc(), cmpLUT_op.getType(), sub_operands);
                    llvm::SmallVector<Value> falseMul_operands;
                    falseMul_operands.push_back(sub_op.getResult());
                    falseMul_operands.push_back(falseValue);
                    auto falseMul_op = rewriter.create<LWEMulOp>(sub_op.getLoc(), sub_op.getType(), falseMul_operands);
                    llvm::SmallVector<Value> finalAdd_operands;
                    finalAdd_operands.push_back(trueMul_op.getResult());
                    finalAdd_operands.push_back(falseMul_op.getResult());
                    auto finalAdd_op = rewriter.create<LWEAddOp>(falseMul_op.getLoc(), falseMul_op.getType(), finalAdd_operands);
                    auto finalLUT_op = rewriter.create<FHELUTOp>(finalAdd_op.getLoc(), finalAdd_op.getType(), finalAdd_op.getResult());

                    op->replaceAllUsesWith(finalLUT_op);
                    rewriter.eraseOp(op);
                    c_op->replaceAllUsesWith(cmpLUT_op);
                    rewriter.eraseOp(c_op);
                }
            }
        }

        return success();
    }
};

// Replace the previous ArithCmpFPattern, because we
// need to optimize (eliminate) SelectOp and CmpFOp in order
LogicalResult FHECmpOperation(IRRewriter &rewriter, MLIRContext *context, FHECmpOp c_op)
{
    rewriter.setInsertionPoint(c_op);

    CmpFPredicate predicate = c_op.predicate();
    Value cmp_lhs = c_op.lhs();
    Value cmp_rhs = c_op.rhs();
    llvm::SmallVector<Value> c_operands;
    c_operands.push_back(cmp_lhs);
    c_operands.push_back(cmp_rhs);

    auto cmpSub_op = rewriter.create<LWESubOp>(c_op.getLoc(), c_op.getType(), c_operands);

    if (predicate == CmpFPredicate::OGT) {
        // llvm::outs() << "GT Branch\n";
        auto cmpLUT_op = rewriter.create<FHELUTForGTOp>(cmpSub_op.getLoc(), cmpSub_op.getType(), cmpSub_op.getResult());
        c_op->replaceAllUsesWith(cmpLUT_op);
        c_op.erase();
    }
    else if (predicate == CmpFPredicate::OLT) {
        auto cmpLUT_op = rewriter.create<FHELUTForLTOp>(cmpSub_op.getLoc(), cmpSub_op.getType(), cmpSub_op.getResult());
        c_op->replaceAllUsesWith(cmpLUT_op);
        c_op.erase();
    }
    else 
        return failure();

    return success();
}

// Now, we eliminate scf::IfOp in branch pass
// class ScfIfPattern final : public OpConversionPattern<scf::IfOp>
// {
// public:
//     using OpConversionPattern<scf::IfOp>::OpConversionPattern;

//     LogicalResult matchAndRewrite(
//         scf::IfOp op, typename scf::IfOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
//     {
//         // Value condition = op.getCondition();
//         // auto castOp = rewriter.create<UIToFPOp>(op.getLoc(), op.getType(0), condition);
//         // Value castCondition = castOp.getResult();

//         // auto &thenRegion = op.getThenRegion();
//         // auto *thenBlock = &thenRegion.front();
//         // auto thenOps = op.thenBlock()->without_terminator();
//         // // auto thenSize = thenOps->size();
//         // for (auto thenOperation = thenOps.begin(); thenOperation != thenOps.end(); ++thenOperation) {
//         //     thenOperation->print(llvm::outs());
//         //     auto operationName = thenOperation->getName();
//         //     // auto thenOp = thenOperation->
//         // }
//         // // auto thenBlock = op.thenBlock();
//         // // auto &thenOps = thenBlock->getOperations();
//         // // for (auto thenOp = *thenBlock.begin(); thenOp != *thenBlock.end(); thenOp++) {

//         // // }
//         Value condition = op.getCondition();
//         Value LWECondition;
//         if (FHEMaterializeOp mat_op = condition.getDefiningOp<FHEMaterializeOp>()) {
//             Value matInput = mat_op.input();
//             if (auto st = matInput.getType().dyn_cast<LWECipherType>()) {
//                 LWECondition = matInput;
//             }
//             else {
//                 llvm::outs() << "The input type of MaterializeOp in SCFIfPattern is not LWECipherType";
//                 return failure();
//             }
//         }
//         else {
//             llvm::outs() << "No MaterializeOp before IfOp condition";
//             return failure();
//         }
//         Block *thenBlock = op.thenBlock();
//         Block *elseBlock = op.elseBlock();

//         int numThenOps = std::distance(thenBlock->begin(), thenBlock->end());
//         int numElseOps = std::distance(elseBlock->begin(), elseBlock->end());
        

//         // 1. For Else Branch
//         Value elseResult;
//         if (numElseOps == 1) {
//             // auto elseYieldOp = elseBlock->getOperations().front();
//             auto elseYieldOp = op.elseYield();
//             Value elseYieldResult = elseYieldOp.getResults().front();
//             if (FHEMaterializeOp mat_op = elseYieldResult.getDefiningOp<FHEMaterializeOp>()) {
//                 elseResult = mat_op.input();
//             }
//         }

//         // 2. For Then Branch
//         Value thenResult;
//         for (Operation &thenOp : thenBlock->getOperations()) {
//             if (isa<scf::YieldOp>(thenOp)) {
//                 scf::YieldOp thenYieldOp = dyn_cast<scf::YieldOp>(thenOp);
//                 Value thenYieldResult = thenYieldOp.getResults().front();
//                 if (FHEMaterializeOp mat_op = thenYieldResult.getDefiningOp<FHEMaterializeOp>()) {
//                     thenResult = mat_op.input();
//                 }
//             }
//         }

//         if (elseResult.use_empty() || thenResult.use_empty()) 
//             return failure();

//         // for (Operation & thenop : thenBlock->getOperations()) {
//         //     llvm::outs() << thenop << "\n";
//         // }
//         // Copy Thenbranch and Elsebranch outside IfOp


//         Value result = op.getResult(0);
//         // Value thenResult = op.thenYield().getResults().front();
//         // Value elseResult = op.elseYield().getResults().front();

//         llvm::SmallVector<Value> trueMul_operands;
//         trueMul_operands.push_back(LWECondition);
//         trueMul_operands.push_back(thenResult);
//         auto trueMul_op = rewriter.create<LWEMulOp>(op.getLoc(), LWECondition.getType(), trueMul_operands);

//         llvm::outs() << trueMul_op;
        
//         FloatAttr oneAttr = FloatAttr::get(Float32Type::get(rewriter.getContext()), 1.);
//         // oneAttr.get(Float32Type::get(rewriter.getContext()), 1.);
//         llvm::outs() << oneAttr.getValue().convertToFloat();
//         auto encode_op = rewriter.create<FHEEncodeOp>(trueMul_op.getLoc(), 
//                                             PlainType::get(rewriter.getContext()), oneAttr);
//         llvm::SmallVector<Value> sub_operands;
//         sub_operands.push_back(encode_op.getResult());
//         sub_operands.push_back(LWECondition);
//         auto sub_op = rewriter.create<LWESubOp>(encode_op.getLoc(), LWECondition.getType(), sub_operands);
//         llvm::SmallVector<Value> falseMul_operands;
//         falseMul_operands.push_back(sub_op.getResult());
//         falseMul_operands.push_back(elseResult);
//         auto falseMul_op = rewriter.create<LWEMulOp>(sub_op.getLoc(), sub_op.getType(), falseMul_operands);
//         llvm::SmallVector<Value> finalAdd_operands;
//         finalAdd_operands.push_back(trueMul_op.getResult());
//         finalAdd_operands.push_back(falseMul_op.getResult());
//         auto finalAdd_op = rewriter.create<LWEAddOp>(falseMul_op.getLoc(), falseMul_op.getType(), finalAdd_operands);
//         auto finalLUT_op = rewriter.create<FHELUTOp>(finalAdd_op.getLoc(), finalAdd_op.getType(), finalAdd_op.getResult());
//         auto finalMat_op = rewriter.create<FHEMaterializeOp>(finalLUT_op.getLoc(), result.getType(), finalLUT_op.result());
        
//         result.replaceAllUsesWith(finalMat_op);

//         func::FuncOp func_op = op.getOperation()->getParentOfType<func::FuncOp>();
//         func_op.print(llvm::outs());
//         llvm::outs() << "\n\n\n";
//         // thenBlock->moveBefore(op);
//         // elseBlock->moveBefore(op);
//         op.erase();
//         // rewriter.replaceOpWithNewOp<FHEMaterializeOp>(op, result.getType(), finalLUT_op.getResult());

//         return success();
//     }
// };

// unfold branches caused by comparison operations 
// in the input program 
void BranchUnfoldPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, HEIRDialect, ArithmeticDialect>();
    target.addLegalOp<ModuleOp>();
    // target.addIllegalOp<CmpFOp>();
    // target.addIllegalOp<SelectOp>();
    target.addIllegalOp<FHESelectOp>();
    // target.addIllegalOp<scf::IfOp>();
    IRRewriter rewriter(&getContext());
    
    // 1. For FHECompOp and FHESelectOp exist togethor
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<FHESelectPattern>(patterns.getContext());
    
    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();

    // 2. For single FHECompOp exist
    auto &block = getOperation()->getRegion(0).getBlocks().front();

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        if(f.walk([&](Operation *op)
                {
        if (FHECmpOp cmp_op = llvm::dyn_cast_or_null<FHECmpOp>(op)) {
            if (FHECmpOperation(rewriter, &getContext(), cmp_op).failed()) {
                return WalkResult::interrupt();
            }
        }
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
    }
}