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

#include "heir/Passes/nary/Nary.h"
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heir;

void NaryPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<
        heir::HEIRDialect, mlir::AffineDialect, func::FuncDialect, mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
}

// LWEAdd aggregation
void collapseAdd(heir::LWEAddOp &op, IRRewriter &rewriter)
{
    for (auto &use : llvm::make_early_inc_range(op.output().getUses()))
    {
        if (auto use_add = llvm::dyn_cast<heir::LWEAddOp>(*use.getOwner()))
        {
            rewriter.setInsertionPointAfter(use_add.getOperation());
            llvm::SmallVector<Value, 4> new_operands;
            for (auto s : use_add.x())
            {
                if (s != op.output())
                {
                    new_operands.push_back(s);
                }
            }
            for (auto s : op.x())
            {
                new_operands.push_back(s);
            }
            auto new_add = rewriter.create<heir::LWEAddOp>(use_add->getLoc(), use_add.output().getType(), new_operands);
            use_add.replaceAllUsesWith(new_add.getOperation());
        }
    }
}

// LWESub aggregation
void collapseSub(heir::LWESubOp &op, IRRewriter &rewriter)
{
    for (auto &use : llvm::make_early_inc_range(op.output().getUses()))
    {
        if (auto use_sub = llvm::dyn_cast<heir::LWESubOp>(*use.getOwner()))
        {
            rewriter.setInsertionPointAfter(use_sub.getOperation());
            llvm::SmallVector<Value, 4> new_operands;
            for (auto s : use_sub.x())
            {
                if (s != op.output())
                {
                    new_operands.push_back(s);
                }
            }
            for (auto s : op.x())
            {
                new_operands.push_back(s);
            }
            auto new_sub = rewriter.create<heir::LWESubOp>(use_sub->getLoc(), use_sub.output().getType(), new_operands);
            use_sub.replaceAllUsesWith(new_sub.getOperation());
        }
    }
}

// LWEMul aggregation
void collapseMul(heir::LWEMulOp &op, IRRewriter &rewriter)
{
    for (auto &use : llvm::make_early_inc_range(op.output().getUses()))
    {
        if (auto use_mul = llvm::dyn_cast<heir::LWEMulOp>(*use.getOwner()))
        {
            rewriter.setInsertionPointAfter(use_mul.getOperation());
            llvm::SmallVector<Value, 4> new_operands;
            for (auto s : use_mul.x())
            {
                if (s != op.output())
                {
                    new_operands.push_back(s);
                }
            }
            for (auto s : op.x())
            {
                new_operands.push_back(s);
            }
            auto new_mul =
                rewriter.create<heir::LWEMulOp>(use_mul->getLoc(), use_mul.output().getType(), new_operands);
            use_mul.replaceAllUsesWith(new_mul.getOperation());
        }
    }
}

// add/sub/mult operation aggragation to get ready for batching optimizatins 
void NaryPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect>();
    target.addIllegalOp<AffineForOp>();

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());


    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<arith::SubIOp>()))
        {
            rewriter.setInsertionPointAfter(op.getOperation());
            llvm::SmallVector<Value, 2> operands = { op.getLhs(), op.getRhs() };
            Value value = rewriter.create<heir::LWESubOp>(op.getLoc(), op.getResult().getType(), operands);
            op.replaceAllUsesWith(value);
            rewriter.eraseOp(op.getOperation());
        }
    }

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<arith::MulIOp>()))
        {
            rewriter.setInsertionPointAfter(op.getOperation());
            llvm::SmallVector<Value, 2> operands = { op.getLhs(), op.getRhs() };
            Value value = rewriter.create<heir::LWEMulOp>(op.getLoc(), op.getResult().getType(), operands);
            op.replaceAllUsesWith(value);
            rewriter.eraseOp(op.getOperation());
        }
    }

    // Now, go through and actually start collapsing the operations!
    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<heir::LWEAddOp>()))
        {
            collapseAdd(op, rewriter);
        }
    }

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<heir::LWESubOp>()))
        {
            collapseSub(op, rewriter);
        }
    }

    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<heir::LWEMulOp>()))
        {
            collapseMul(op, rewriter);
        }
    }

    // TODO: Finally, clean up ops without uses (hard to do during above loop because of iter invalidation)
    //  HACK: just canonicalize again for now ;)
}

// TODO: NOTE WE MUST ADD CSE MANUALLY!