// Author: Zian Zhao
#include <queue>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "heir/Passes/slot2coeff/SlotToCoeff.h"

using namespace mlir;
using namespace heir;

void SlotToCoeffPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<heir::HEIRDialect,
                  mlir::AffineDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect>();
}

// coefficient-encoding optimization for inner_product and euclid_dist
LogicalResult batchInnerProductOperation(IRRewriter &rewriter, MLIRContext *context, LWEAddOp op)
{
    int numOperands = op.getNumOperands();
    // Now we only support batching for vector length >= 4
    if (numOperands <= 2) {
        return success();
    }

    // Check if the operation satisfies the structure of inner_product
    // 1. rotate
    int numRotateOp = 0;
    llvm::SmallVector<Value> rotOperands;
    std::vector<int32_t> rotIndices;
    Value originCipher;
    for (Value it : op.getOperands()) {
        if (FHERotateOp rot_op = it.getDefiningOp<FHERotateOp>()) {
            numRotateOp++;
            rotOperands.push_back(it);
            rotIndices.push_back(rot_op.i());
            if (numRotateOp > 1) {
                if (it.getDefiningOp<FHERotateOp>().cipher() != rotOperands[0].getDefiningOp<FHERotateOp>().cipher()) break;
            }
        }
    }
    if (numRotateOp != numOperands - 1)
        return success();
    // If indices satisfy {-1, -2, -3, ...}
    auto vector_length = op.output().getType().dyn_cast_or_null<RLWECipherType>().getSize();
    for (int i = -1; i > -vector_length; i--) {
        auto it = std::find(rotIndices.begin(), rotIndices.end(), i);
        if (it == rotIndices.end()) return success();
    }



    LWEMulOp rlweMulOp;
    for (Value it : op.getOperands()) {
        if (LWEMulOp mul_op = it.getDefiningOp<LWEMulOp>()) {
            if (it != rotOperands[0].getDefiningOp<FHERotateOp>().cipher()) break;
            originCipher = it;
            rlweMulOp = mul_op;
        }
    }

    op->replaceAllUsesWith(rlweMulOp);
    rewriter.setInsertionPoint(rlweMulOp);
    // auto newMulOp = rewriter.create<RLWEMulOp>(rlweMulOp.getLoc(), rlweMulOp.getType(), rlweMulOp.getOperands());
    auto new_mul_op = rewriter.replaceOpWithNewOp<RLWEMulOp>(rlweMulOp, rlweMulOp.getType(), rlweMulOp.getOperands());

    for (OpOperand & use : new_mul_op->getUses()) {
        Operation *userOp = use.getOwner();
        if (FHEExtractfinalOp ex_op = dyn_cast<FHEExtractfinalOp>(userOp)) {
            Value input = ex_op.vector();
            auto vecSize = input.getType().dyn_cast<RLWECipherType>().getSize();
            auto indexAttr = IntegerAttr::get(IndexType::get(rewriter.getContext()), vecSize - 1);
            ex_op.colAttr(indexAttr);
        }
    }

    // For EuclidDist, compputing euclid_dist with three multiplications
    if (rlweMulOp.getOperand(0) == rlweMulOp.getOperand(1)) {
        if (LWESubOp subOp = rlweMulOp.getOperand(0).getDefiningOp<LWESubOp>()) {
            // llvm::outs() << "hello";
            Value vec1 = subOp.getOperand(0);
            Value vec2 = subOp.getOperand(1);

            rewriter.setInsertionPoint(subOp);
            SmallVector<Value> v1MulOperands;
            v1MulOperands.push_back(vec1);
            v1MulOperands.push_back(vec1);
            auto v1MulOp = rewriter.create<RLWEMulOp>(subOp.getLoc(), vec1.getType(), v1MulOperands);
            SmallVector<Value> v2MulOperands;
            v2MulOperands.push_back(vec2);
            v2MulOperands.push_back(vec2);
            auto v2MulOp = rewriter.create<RLWEMulOp>(subOp.getLoc(), vec2.getType(), v2MulOperands);
            SmallVector<Value> v1v2MulOperands;
            v1v2MulOperands.push_back(vec1);
            v1v2MulOperands.push_back(vec2);
            auto v12MulOp = rewriter.create<RLWEMulOp>(subOp.getLoc(), vec1.getType(), v1v2MulOperands);
            SmallVector<Value> v12DoubleOperands;
            v12DoubleOperands.push_back(v12MulOp.output());
            v12DoubleOperands.push_back(v12MulOp.output());
            auto v12DoubleOp = rewriter.create<LWEAddOp>(subOp.getLoc(), vec1.getType(), v12DoubleOperands);
            SmallVector<Value> v1v2AddOperands;
            v1v2AddOperands.push_back(v1MulOp.output());
            v1v2AddOperands.push_back(v2MulOp.output());
            auto v1v2AddOp = rewriter.create<LWEAddOp>(subOp.getLoc(), vec1.getType(), v1v2AddOperands);
            SmallVector<Value> finalOperands;
            finalOperands.push_back(v1v2AddOp.output());
            finalOperands.push_back(v12DoubleOp.output());
            auto finalOp = rewriter.create<LWESubOp>(subOp.getLoc(), vec1.getType(), finalOperands);

            new_mul_op->replaceAllUsesWith(finalOp);

        }
    }


    // rewriter.eraseOp(op);
    
    return success();
}

// several pattern can be optimized via coefficient-encoding instead of 
// slot-encoding, e.g., InnerProduct and EuclidDist
// TODO: Add a new optimization for convolution
void SlotToCoeffPass::runOnOperation()
{ 
  // Get the (default) block in the module's only region:
  auto &block = getOperation()->getRegion(0).getBlocks().front();
  IRRewriter rewriter(&getContext());

  for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
  {
    if (f.walk([&](Operation *op)
               {

      if (LWEAddOp add_op = llvm::dyn_cast_or_null<LWEAddOp>(op)) {
        if (batchInnerProductOperation(rewriter, &getContext(), add_op).failed())
          return WalkResult::interrupt();
      }
      return WalkResult(success()); })
            .wasInterrupted())
      signalPassFailure();
  }
}