/*
Authors: HECO
Modified by Zian Zhao
Copyright:
Copyright (c) 2020 ETH Zurich.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <queue>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "heir/Passes/batching/Batching.h"

using namespace mlir;
using namespace heir;

void BatchingPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<heir::HEIRDialect,
                  mlir::AffineDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect>();
}

typedef Value VectorValue;
typedef llvm::SmallMapVector<VectorValue, Value, 1> scalarBatchingMap;

template <typename OpType>
LogicalResult batchArithmeticOperation(IRRewriter &rewriter, MLIRContext *context, OpType op)
{
    // We care only about ops that return scalars, assuming others are already "SIMD-compatible"
    if (auto result_st = op.getType().template dyn_cast_or_null<LWECipherType>())
    {
        /// Target Slot (-1 => no target slot required)
        int target_slot = -1;

        // TODO: as a basic fix, we only search one level deep for now.
        //  This prevents the issue mentioned above but of course might prevent optimizations for complex code
        for (auto u : op->getUsers())
        {
            if (FHEExtractfinalOp ex_op = dyn_cast_or_null<FHEExtractfinalOp>(u))
            {
                // We later want this value to be in slot i??
                target_slot = ex_op.col().cast<IntegerAttr>().getValue().getLimitedValue(INT32_MAX);
                break;
            }
            else if (FHEInsertfinalOp ins_op = dyn_cast_or_null<FHEInsertfinalOp>(u))
            {
                target_slot = ins_op.col().cast<IntegerAttr>().getValue().getLimitedValue(INT32_MAX);
                break;
            }
            else if (auto ret_op = dyn_cast_or_null<func::ReturnOp>(u))
            {
                if (ret_op->getOperandTypes().front().template isa<LWECipherType>())
                    // we eventually want this as a scalar, which means slot 0
                    target_slot = 0;
                break;
            }
        }

        // instead of just picking the first target slot we see, we check if we can find 0
        if (target_slot == -1)
        {
            for (auto it = op->operand_begin(); it != op.operand_end(); ++it)
            {
                if (auto st = (*it).getType().template dyn_cast_or_null<LWECipherType>())
                {
                    // scalar-type input that needs to be converted
                    if (FHEExtractfinalOp ex_op = (*it).template getDefiningOp<FHEExtractfinalOp>())
                    {
                        // auto i = (int)ex_op.i().getLimitedValue();
                        auto i = (int)ex_op.col().cast<IntegerAttr>().getValue().getLimitedValue(INT32_MAX);
                        if (target_slot == -1)
                            target_slot = i;
                    }
                }
            }
        }

        // new op
        rewriter.setInsertionPointAfter(op);
        auto new_op = rewriter.create<OpType>(op.getLoc(), op.getType(), op->getOperands());
        rewriter.setInsertionPoint(new_op); // otherwise, any operand transforming ops will be AFTER the new op

        // find the maximum size of vector involved
        int max_size = -1;
        for (auto o : new_op.getOperands())
        {
            // o.print(llvm::outs());
            if (auto bst = o.getType().template dyn_cast_or_null<LWECipherVectorType>())
            {
                max_size = std::max(max_size, bst.getSize());
            }
            else if (auto st = o.getType().template dyn_cast_or_null<LWECipherType>())
            {
                // scalar-type input that will be converted
                if (FHEExtractfinalOp ex_op = o.template getDefiningOp<FHEExtractfinalOp>())
                {
                    auto bst = ex_op.vector().getType().dyn_cast_or_null<LWECipherVectorType>();
                    assert(bst && "Extractfinal must be applied to LWECipherVector");
                    max_size = std::max(max_size, bst.getSize());
                }
            }
        }

        // convert all operands from scalar to batched
        for (auto it = new_op->operand_begin(); it != new_op.operand_end(); ++it)
        {
            if (auto bst = (*it).getType().template dyn_cast_or_null<LWECipherVectorType>())
            {
                // Check if it needs to be resized
                if (bst.getSize() < max_size)
                {
                    // auto resized_type =
                    //     fhe::BatchedSecretType::get(rewriter.getContext(), bst.getPlaintextType(), max_size);
                    auto resized_type = 
                          LWECipherVectorType::get(rewriter.getContext(), bst.getPlaintextType(), max_size);
                    auto resized_o = rewriter.create<FHEMaterializeOp>(op.getLoc(), resized_type, *it);
                    rewriter.replaceOpWithIf((*it).getDefiningOp(), { resized_o }, [&](OpOperand &operand) {
                        return operand.getOwner() == new_op;
                    });
                }
            }
            else if (auto st = (*it).getType().template dyn_cast_or_null<LWECipherType>())
            {
                // scalar-type input that needs to be converted
                if (FHEExtractfinalOp ex_op = (*it).template getDefiningOp<FHEExtractfinalOp>())
                {
                    // Check if it needs to be resized
                    if (auto bst = ex_op.vector().getType().template dyn_cast_or_null<LWECipherVectorType>())
                    {
                        if (bst.getSize() < max_size)
                        {
                            auto resized_type = 
                                LWECipherVectorType::get(rewriter.getContext(), bst.getPlaintextType(), max_size);
                            auto cur_ip = rewriter.getInsertionPoint();
                            rewriter.setInsertionPoint(ex_op);
                            Value resized_o =
                                rewriter.create<FHEMaterializeOp>(ex_op.getLoc(), resized_type, ex_op.vector());
                            auto resized_ex_op = rewriter.create<FHEExtractfinalOp>(
                                ex_op.getLoc(), ex_op.getType(), resized_o, ex_op.col(), Attribute());
                            rewriter.replaceOpWithIf(ex_op, { resized_ex_op }, [&](OpOperand &operand) {
                                return operand.getOwner() == new_op;
                            });
                            ex_op = resized_ex_op;
                            rewriter.setInsertionPoint(&*cur_ip);
                        }
                    }

                    // instead of using the extract op, issue a rotation instead
                    auto i = (int)ex_op.col().cast<IntegerAttr>().getValue().getLimitedValue(INT32_MAX);
                    if (target_slot == -1) // no other target slot defined yet, let's make this the target
                        target_slot = i; // we'll rotate by zero, but that's later canonicalized to no-op anyway
                    auto rotate_op = rewriter.create<FHERotateOp>(ex_op.getLoc(), ex_op.vector().getType(), ex_op.vector(), target_slot - i);
                    rewriter.replaceOpWithIf(
                        ex_op, { rotate_op }, [&](OpOperand &operand) { return operand.getOwner() == new_op; });
                }
                // HECO uses fhe::ConstOp to cast a plaintext into a secret value
                else if (auto c_op = (*it).template getDefiningOp<FHEEncodeOp>())
                {
                    Type resized_type = c_op.getType();
                    if (auto bst = c_op.getType().template dyn_cast_or_null<LWECipherVectorType>())
                    {
                        resized_type =
                            LWECipherVectorType::get(rewriter.getContext(), bst.getPlaintextType(), max_size);
                    }
                    else if (auto st = c_op.getType().template dyn_cast_or_null<LWECipherType>())
                    {
                        resized_type =
                            LWECipherVectorType::get(rewriter.getContext(), st.getPlaintextType(), max_size);
                    }
                    else
                    {
                        assert(false && "This should not be possible");
                    }
                    auto new_cst = rewriter.template create<FHEEncodeOp>(c_op.getLoc(), resized_type, c_op.message());
                    rewriter.replaceOpWithIf(
                        c_op, { new_cst }, [&](OpOperand &operand) { return operand.getOwner() == new_op; });
                }
                else
                {
                    emitWarning(
                        new_op.getLoc(), "Encountered unexpected (non batchable) defining op for secret operand "
                                         "while trying to batch.");
                    return failure();
                }
            }
            else
            {
                // non-secret input, which we can always transform as needed -> no action needed now
            }
        }
        // re-create the op to get correct type inference
        // TODO: avoid this by moving op creation from before operands to after
        // auto newer_op = rewriter.create<OpType>(new_op.getLoc(), LWECipherVectorType)
        auto newer_op = rewriter.create<OpType>(new_op.getLoc(), new_op->getOperands()[0].getType(), new_op->getOperands());
        rewriter.eraseOp(new_op);
        new_op = newer_op;


        // Now create a scalar again by creating an extract, preserving type constraints
        rewriter.setInsertionPointAfter(new_op);
        auto res_ex_op = rewriter.create<FHEExtractfinalOp>(
            op.getLoc(), op.getType(), new_op.getResult(), rewriter.getIndexAttr(target_slot), Attribute());
        op->replaceAllUsesWith(res_ex_op);

        // Finally, remove the original op
        rewriter.eraseOp(op);
    }
    return success();
}


LogicalResult RotateOpToRLWEOperation(IRRewriter &rewriter, MLIRContext *context, FHERotateOp op, TypeConverter typeConverter)
{
    rewriter.setInsertionPoint(op);

    auto dstType = typeConverter.convertType(op.getType());
    if (!dstType)
        return failure();


    Value operand = op.getOperand();
    auto operandDstType = typeConverter.convertType(operand.getType());
    Value new_operand = typeConverter.materializeTargetConversion(rewriter, op.getLoc(), operandDstType, operand);
    auto rotIndex = op.i();

    
    rewriter.replaceOpWithNewOp<FHERotateOp>(op, dstType, new_operand, rotIndex);

    return success();
    
}

// Batching many thousands of values into a single vector-like ciphertext.
// Ready to convert LWE ciphertext to RLWE ciphertext
void BatchingPass::runOnOperation()
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

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if(f.walk([&](Operation *op)
                {
        if (LWESubOp sub_op = llvm::dyn_cast_or_null<LWESubOp>(op)) {
            if (batchArithmeticOperation<LWESubOp>(rewriter, &getContext(), sub_op).failed())
            return WalkResult::interrupt();
        } else if (LWEAddOp add_op = llvm::dyn_cast_or_null<LWEAddOp>(op)) {
            if (batchArithmeticOperation<LWEAddOp>(rewriter, &getContext(), add_op).failed())
            return WalkResult::interrupt();
        } else if (LWEMulOp mul_op = llvm::dyn_cast_or_null<LWEMulOp>(op)) {
            if (batchArithmeticOperation<LWEMulOp>(rewriter, &getContext(), mul_op).failed())
            return WalkResult::interrupt();
        }
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
        // f.print(llvm::outs());
    }


    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // We must translate in order of appearance for this to work, so we walk manually
        if (f.walk([&](Operation *op)
                {
        
        if (FHERotateOp rot_op = llvm::dyn_cast_or_null<FHERotateOp>(op)) {
            if (RotateOpToRLWEOperation(rewriter, &getContext(), rot_op, type_converter).failed())
            return WalkResult::interrupt();
        }
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
    }


}