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
#include <iostream>
#include <memory>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "heir/Passes/unroll/UnrollLoop.h"

using namespace mlir;
using namespace arith;

void UnrollLoopPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<ArithmeticDialect>();
    registry.insert<AffineDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<heir::HEIRDialect>();
}

void unrollLoop(AffineForOp &op, IRRewriter &rewriter)
{
    // First, let's recursively unroll all nested loops:
    for (auto nested_loop : op.getOps<AffineForOp>())
    {
        unrollLoop(nested_loop, rewriter);
    }

    // TODO: Fix MLIR issues in mlir/Transforms/LoopUtils.h where the typedef for FuncOp is messing with the fwd
    // declaration of FuncOp
    if (loopUnrollFull(op).failed())
    {
        emitError(op.getLoc(), "Failed to unroll loop");
    }
}

// Unroll all the for loops in the input program,
// but we do not actually use this pass because 
// perhaps there exists multiple nested for loops
// for now, we use the built-in "affine-loop-unroll" pass
void UnrollLoopPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<AffineDialect, func::FuncDialect, tensor::TensorDialect, scf::SCFDialect, ArithmeticDialect>();
    target.addLegalDialect<heir::HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<scf::IfOp>();
    target.addLegalOp<scf::ForOp>();
    // target.addIllegalOp<AffineForOp>();

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    IRRewriter rewriter(&getContext());

    // TODO: There's likely a much better way to do this that's not this kind of manual walk!
    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<AffineForOp>()))
        {
            unrollLoop(op, rewriter);
        }
    }
    
}