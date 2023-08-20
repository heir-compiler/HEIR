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

#include "heir/IR/FHE/HEIRDialect.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heir;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heir/IR/FHE/HEIRTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "heir/IR/FHE/HEIR.cpp.inc"

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult heir::FHEMaterializeOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (auto m_op = input().getDefiningOp<heir::FHEMaterializeOp>()) {
        if (m_op.input().getType() == result().getType())
            return m_op.input();
        if (auto mm_op = m_op.input().getDefiningOp<heir::FHEMaterializeOp>()) {
            if (mm_op.input().getType() == result().getType())
                return mm_op.input();
        }
    }
    else if (input().getType() == result().getType()) {
        return input();
    }
    return {};
    

}

/// simplify rotate(cipher, 0) to cipher
::mlir::OpFoldResult heir::FHERotateOp::fold(::llvm::ArrayRef<::mlir::Attribute> operands)
{
    if (i() == 0)
        return cipher();

    return {};
}

//===----------------------------------------------------------------------===//
// FHE dialect definitions
//===----------------------------------------------------------------------===//
#include "heir/IR/FHE/HEIRDialect.cpp.inc"
void HEIRDialect::initialize()
{
    // Registers all the Types into the FHEDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heir/IR/FHE/HEIRTypes.cpp.inc"
        >();

    // Registers all the Operations into the FHEDialect class
    addOperations<
#define GET_OP_LIST
#include "heir/IR/FHE/HEIR.cpp.inc"
        >();
}