#include "heir/Passes/genparm/GenParm.h"
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

void GenerateParmPass::getDependentDialects(mlir::DialectRegistry &registry) const 
{
    registry.insert<func::FuncDialect>();
    registry.insert<HEIRDialect>();
}

// Generate the level of each ciphertext Value
// TODO: Does not integrate to the IR pass
size_t determinBootstrapLevel(Value bootOutput, unsigned int curLine)
{
    std::vector<size_t> uses_list;
    if (bootOutput.getUses().empty())
        return 0;
    for (OpOperand &u : bootOutput.getUses()) {
        size_t localLevel = 0;
        auto *subOp = u.getOwner();
        
        // 1. Result is stored to a matrix/vector, find the next extract op 
        if (auto op = dyn_cast<FHEInsertfinalOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";

            APInt insRow = op.rowAttr().cast<IntegerAttr>().getValue();
            APInt insCol = op.colAttr().cast<IntegerAttr>().getValue();
            Value insVector = op.memref();
            unsigned int insLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            auto curBlock = subOp->getBlock();
            
            // TODO: maybe we cannot call determinateBootstrapLevel() recurisively??
            FHEExtractfinalOp nearExtOp;
            unsigned int nearExtDist = 99999;
            auto extractOps = curBlock->getOps<FHEExtractfinalOp>();
            for (FHEExtractfinalOp extractOp : extractOps) {
                APInt extRow = extractOp.rowAttr().cast<IntegerAttr>().getValue();
                APInt extCol = extractOp.colAttr().cast<IntegerAttr>().getValue();
                Value extVector = extractOp.vector();
                unsigned int extLine = LocationAttr(extractOp.getLoc()).cast<FileLineColLoc>().getLine();
                // Same slot in vector
                if (extRow == insRow && extCol == insCol && extVector == insVector 
                        && extLine > insLine && extLine - insLine < nearExtDist) {
                    nearExtOp = extractOp;
                    nearExtDist = extLine - insLine;
                }
            }
            FHEVectorLoadfinalOp nearVecLoadOp;
            unsigned int nearVecLoadDist = 99999;
            auto vecExtOps = curBlock->getOps<FHEVectorLoadfinalOp>();
            for (FHEVectorLoadfinalOp vecExtOp : vecExtOps) {
                APInt extRow = vecExtOp.index().cast<IntegerAttr>().getValue();
                Value extVector = vecExtOp.memref();
                unsigned int extLine = LocationAttr(vecExtOp.getLoc()).cast<FileLineColLoc>().getLine();
                if (extRow == insRow && extVector == insVector 
                        && extLine > insLine && extLine - insLine < nearVecLoadDist) {
                    nearVecLoadOp = vecExtOp;
                    nearVecLoadDist = extLine - insLine;
                }
            }
            // Check if another store Op override the current result
            FHEInsertfinalOp nearInsertOp;
            unsigned int nearInsertDist = 99999;
            auto insertOps = curBlock->getOps<FHEInsertfinalOp>();
            for (FHEInsertfinalOp insertOp : insertOps) {
                APInt insertRow = insertOp.rowAttr().cast<IntegerAttr>().getValue();
                APInt insertCol = insertOp.colAttr().cast<IntegerAttr>().getValue();
                Value insertVector = insertOp.memref();
                unsigned int insertLine = LocationAttr(insertOp.getLoc()).cast<FileLineColLoc>().getLine();
                if (insertRow == insRow && insertCol == insCol && insertVector == insVector 
                        && insertLine > insLine && insertLine - insLine < nearInsertDist) {
                    nearInsertOp = insertOp;
                    nearInsertDist = insertLine - insLine;
                }
            }
            // Check if another store Op override the current result
            if (nearInsertDist < std::min(nearVecLoadDist, nearExtDist)) {
                llvm::outs() << "This FHEInsertfinalOp has been overrided by another Insert Op\n";
                break;
            }
            else if (nearVecLoadDist < nearExtDist) {
                // llvm::outs() << "VecLoadOp is the next Op\n";
                // nearVecLoadOp->getName().print(llvm::outs());
                // nearVecLoadOp->getLoc().print(llvm::outs());
                // llvm::outs() << "\n";

                Value subOutput = nearVecLoadOp.getResult();
                auto subLine = LocationAttr(nearVecLoadOp.getLoc()).cast<FileLineColLoc>().getLine();
                localLevel += determinBootstrapLevel(subOutput, subLine);
            }
            else if (nearVecLoadDist > nearExtDist) {
                // llvm::outs() << "ExtractOp is the next Op\n";
                // nearExtOp->getName().print(llvm::outs());
                // nearExtOp->getLoc().print(llvm::outs());
                // llvm::outs() << "\n";

                Value subOutput = nearExtOp.getResult();
                auto subLine = LocationAttr(nearExtOp.getLoc()).cast<FileLineColLoc>().getLine();
                localLevel += determinBootstrapLevel(subOutput, subLine);
            }
            // No any subsequent Op after the current InsertOp
            else if (nearVecLoadDist == nearExtDist) {
                // subOp->getName().print(llvm::outs());
                // subOp->getLoc().print(llvm::outs());
                // llvm::outs() << "\n";
                llvm::outs() << "!!!!!!!!!!!!!!!!!!!!No subsequent Op!!!!!!!!!!!!\n";
                // break;
            }
        }
        
        // 2. Multiplications 
        else if (auto op = dyn_cast<LWEMulOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";

            localLevel++;
            Value subOutput = op.getResult();
            auto subLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            localLevel += determinBootstrapLevel(subOutput, subLine);
        }
        else if (auto op = dyn_cast<FHEFuncCallOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // llvm::outs() << op.callee();
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";

            if (op.callee().str().find("lut") == 0) {
                // llvm::outs() << "callee name in col " << op.callee().str().find("lut");
                llvm::outs() << "This is a LUT Function in line " 
                << LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine() << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11\n";
                return 0;
            }

            if (op.callee().str().find("inner") == 0) {
                llvm::outs() << "!!!!!!!!!!!!!!!!!!!!!!inner_product!!!!!!!!!!!!!!!!!!\n";
                localLevel++;
            }
            else if (op.callee().str().find("euclid") == 0) {
                llvm::outs() << "!!!!!!!!!!!!!!!!!!!!!!euclid_dist!!!!!!!!!!!!!!!!!!\n";
                localLevel++;
            }
            Value subOutput = op.getResult();
            auto subLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            localLevel += determinBootstrapLevel(subOutput, subLine);
        }

        // 3. Other operators
        else if (auto op = dyn_cast<LWEAddOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";
            Value subOutput = op.getResult();
            auto subLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            localLevel += determinBootstrapLevel(subOutput, subLine);
        }
        else if (auto op = dyn_cast<LWESubOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";
            Value subOutput = op.getResult();
            auto subLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            localLevel += determinBootstrapLevel(subOutput, subLine);
        }
        else if (auto op = dyn_cast<FHEExtractfinalOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";
            Value subOutput = op.getResult();
            auto subLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            localLevel += determinBootstrapLevel(subOutput, subLine);
        }
        else if (auto op = dyn_cast<FHEVectorLoadfinalOp>(subOp)) {
            // subOp->getName().print(llvm::outs());
            // subOp->getLoc().print(llvm::outs());
            // llvm::outs() << "\n";
            Value subOutput = op.getResult();
            auto subLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
            localLevel += determinBootstrapLevel(subOutput, subLine);
        }
        uses_list.push_back(localLevel);
        // else if (auto op = dyn_cast<func::ReturnOp>(subOp))
        //     return localLevel;
        // else return -1; throw error with other undifined Op.
        
    }
    auto finalLevel = *std::max_element(uses_list.begin(), uses_list.end());
    return finalLevel;
}


/// In order to determine the parameters of fb_keys, we need to traverse the 
/// multiplication depth between two bootstrapping Op. However, for a value 
/// generated from LUT, it is often been stored to a vector/matrix firstly
/// and extract(load) later for further computing. And we cannot find out the 
/// relation between a pair of insert/extract through def-use chain.
/// Here, we *assume* heir.extract is always behind heir.store in terms of 
/// FileLineColLoc. But maybe in some complicated programs it is not always 
/// right? Maybe does not work for programs with large for loops
void GenBootParms(FHEFuncCallOp op, IRRewriter &rewriter)
{
    auto funcName = op.callee().str();
    
    // Only operate on TFHE Bootstrapping Op
    if (funcName.find("lut") != std::string::npos) {
        auto curLine = LocationAttr(op.getLoc()).cast<FileLineColLoc>().getLine();
        Value res = op.getResult();
        size_t level = determinBootstrapLevel(res, curLine);

        llvm::outs() << "LUT in Line " << curLine << " has " << level << " level!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";

        // Add an argument (fb_keys modulus) for LUT function
        auto val = APInt(8, level + 1, false);
        // auto typeName = IndexType()
        // auto a = IntegerAttr(mlir::detail::IntegerAttrStorage(IndexType(), val));
        // auto indexOp = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), level + 1);
        // auto newOp = rewriter.replaceOpWithNewOp<FHEFuncCallOp>(op, op.getResult().getType(),
        //                 op.callee(), ArrayAttr(), ArrayAttr(), op.getOperands());
        auto indexOp = rewriter.create<FHEFuncCallOp>(op.getLoc(), op.getResult().getType(),
                                        op.callee(), ArrayAttr(), ArrayAttr(), op.getOperands());
        // rewriter.eraseOp(op);
        indexOp.replaceAllUsesWith(res);
        
        // op.replaceAllUsesWith(newOp.getResult());
    }
    
    
}

// Determine the level of the function arguments
void GenArgLevel(Value arg, IRRewriter &rewriter)
{
    
    unsigned int argLoc = LocationAttr(arg.getLoc()).cast<FileLineColLoc>().getLine();
    size_t level = determinBootstrapLevel(arg, argLoc);
    llvm::outs() << "level = " << level << "\n";
}

void GenerateParmPass::runOnOperation()
{
    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<HEIRDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalDialect<arith::ArithmeticDialect>();

    IRRewriter rewriter(&getContext());
    

    // Get the (default) block in the module's only region:
    auto &block = getOperation()->getRegion(0).getBlocks().front();
    for (auto f : llvm::make_early_inc_range(block.getOps<func::FuncOp>()))
    {
        // Bootstrapping Key Modulus Selection
        for (auto op : llvm::make_early_inc_range(f.getBody().getOps<FHEFuncCallOp>()))
        {
            GenBootParms(op, rewriter);
        }
        // Input cipher level selection
        auto arg = f.getArguments();
        SmallVector<Value> funcArgVec{arg.begin(), arg.end()};
        for (Value ArgI : arg) {
            ArgI.print(llvm::outs());
            ArgI.getType().print(llvm::outs());
            llvm::outs() << "\n";
            // ArgI->cast<Value>();
            GenArgLevel(ArgI, rewriter);
        }
    }
    
}