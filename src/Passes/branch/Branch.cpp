// Author: Zian Zhao
#include <queue>
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/APSInt.h"

#include "heir/Passes/branch/Branch.h"

using namespace mlir;
using namespace arith;
using namespace heir;

void BranchPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
  registry.insert<heir::HEIRDialect,
                  mlir::AffineDialect,
                  ArithmeticDialect,
                  func::FuncDialect,
                  mlir::scf::SCFDialect>();
}

typedef Value VectorValue;
typedef llvm::SmallMapVector<VectorValue, Value, 1> scalarBatchingMap;

// To unfold scf::IfOp due to the data-independent nature of FHE computation
LogicalResult ScfIfOperation(IRRewriter &rewriter, MLIRContext *context, scf::IfOp op)
{
    rewriter.setInsertionPoint(op);

    Value originCondition = op.getCondition();
    
    Block *thenBlock = op.thenBlock();
    Block *elseBlock = op.elseBlock();

    // Copy Operations
    // 1. ThenBranch
    Value thenResult;
    for (Operation &thenOp : thenBlock->getOperations()) {
        if (isa<scf::YieldOp>(thenOp)) {
            scf::YieldOp thenYieldOp = dyn_cast<scf::YieldOp>(thenOp);
            if (std::distance(thenBlock->begin(), thenBlock->end()) == 1)
                thenResult = thenYieldOp.getResults().front();
        }
        else {
            Operation *newOp = thenOp.clone();
            rewriter.insert(newOp);
            thenResult = newOp->getResults().front();
        }
    }
    // 2. ElseBranch
    Value elseResult;
    for (Operation &elseOp : elseBlock->getOperations()) {
        if (isa<scf::YieldOp>(elseOp)) {
            scf::YieldOp elseYieldOp = dyn_cast<scf::YieldOp>(elseOp);
            elseResult = elseYieldOp.getResults().front();
        }
        else {
            Operation *newOp = elseOp.clone();
            rewriter.insert(newOp);
            elseResult = newOp->getResults().front();
        }
    }
    
    // Unfold the two blocks
    rewriter.setInsertionPointAfter(op);
    auto matOp = rewriter.create<FHEMaterializeOp>(op.getLoc(), Float32Type::get(rewriter.getContext()), originCondition);
    auto thenMulOp = rewriter.create<MulFOp>(matOp.getLoc(), matOp.getType(), matOp.getResult(), thenResult);
    APFloat f1(1.0f);
    auto constOp = rewriter.create<ConstantFloatOp>(thenMulOp.getLoc(), f1, Float32Type::get(rewriter.getContext()));
    auto elseSubOp = rewriter.create<SubFOp>(constOp.getLoc(), matOp.getType(), constOp.getResult(), matOp.getResult());
    auto elseMulOp = rewriter.create<MulFOp>(elseSubOp.getLoc(), matOp.getType(), elseSubOp.getResult(), elseResult);
    auto finalAddOp = rewriter.create<AddFOp>(elseMulOp.getLoc(), matOp.getType(), thenMulOp.getResult(), elseMulOp.getResult());
    auto finalLUTOp = rewriter.create<FHELUTOp>(finalAddOp.getLoc(), finalAddOp.getType(), finalAddOp.getResult());

    op->replaceAllUsesWith(finalLUTOp);

    op.erase();


    return success();
    
}

// Unfold the If-Else Blocks in a MLIR function 
void BranchPass::runOnOperation()
{
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        if(f.walk([&](Operation *op)
                {
        if (scf::IfOp if_op = llvm::dyn_cast_or_null<scf::IfOp>(op)) {
            if (ScfIfOperation(rewriter, &getContext(), if_op).failed())
            return WalkResult::interrupt();
        } 
        return WalkResult(success()); })
                .wasInterrupted())
        signalPassFailure();
    }

}