#ifndef HEIR_IR_HEIR_HEIRDIALECT_H
#define HEIR_IR_HEIR_HEIRDIALECT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/CopyOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include the C++ class declaration for this Dialect (no define necessary for this one)
#include "heir/IR/FHE/HEIRDialect.h.inc"

// Include the C++ class (and associated functions) declarations for this Dialect's types
#define GET_TYPEDEF_CLASSES
#include "heir/IR/FHE/HEIRTypes.h.inc"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"

// Include the C++ class (and associated functions) declarations for this Dialect's operations
#define GET_OP_CLASSES
#include "heir/IR/FHE/HEIR.h.inc"

#endif // HEIR_IR_HEIR_HEIRDIALECT_H
