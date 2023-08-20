set(LLVM_TARGET_DEFINITIONS LowerArithToHEIR.td)
mlir_tablegen(LowerArithToHEIR.cpp.inc -gen-rewriters)
add_public_tablegen_target(arith2heir)

set(CMAKE_INCLUDE_CURRENT_DIR ON)


add_heir_conversion_library(HEIRArithToHEIR
        LowerArithToHEIR.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HEIRHEIRDialect
        )