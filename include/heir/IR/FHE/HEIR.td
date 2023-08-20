//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the HEIR dialect.
//
//===----------------------------------------------------------------------===//

#ifndef HEIR_OPS
#define HEIR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/Arithmetic/IR/ArithmeticBase.td"
// include "mlir/Dialect/Arith/IR/ArithBase.td"


// Provide a definition of the 'heir' dialect in the ODS framework so that we
// can define our operations.
def HEIR_Dialect : Dialect {
  let name = "heir";

  let description = [{
      This dialect represents a common abstraction for all RLWE/LWE operations.
  }];

  // let cppNamespace = "::heir::heir";
  let cppNamespace = "::heir";
    
  // We set this bit to generate a declaration of the `materializeConstant`
  // method so that we can materialize constants for our toy operations.
  // let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
}

// def FHECmpPredicateAttr : I64EnumAttr<
//     "FHECmpPredicate", "",
//     [
//       // I64EnumAttrCase<"AlwaysFalse", 0, "false">,
//       I64EnumAttrCase<"OEQ", 1, "oeq">,
//       I64EnumAttrCase<"OGT", 2, "ogt">,
//       I64EnumAttrCase<"OGE", 3, "oge">,
//       I64EnumAttrCase<"OLT", 4, "olt">,
//       I64EnumAttrCase<"OLE", 5, "ole">,
//       I64EnumAttrCase<"ONE", 6, "one">,
//       I64EnumAttrCase<"ORD", 7, "ord">,
//       I64EnumAttrCase<"UEQ", 8, "ueq">,
//       I64EnumAttrCase<"UGT", 9, "ugt">,
//       I64EnumAttrCase<"UGE", 10, "uge">,
//       I64EnumAttrCase<"ULT", 11, "ult">,
//       I64EnumAttrCase<"ULE", 12, "ule">,
//       I64EnumAttrCase<"UNE", 13, "une">,
//       I64EnumAttrCase<"UNO", 14, "uno">,
//       // I64EnumAttrCase<"AlwaysTrue", 15, "true">,
//     ]> {
//   let cppNamespace = "::heir";
// }

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class HEIR_Op<string mnemonic, list<Trait> traits = []> :
    Op<HEIR_Dialect, mnemonic, !listconcat(traits, [])> {
  
  let assemblyFormat = [{
    `(` operands `)` attr-dict `:`  `(` type(operands) `)` `->` type(results)
  }];
}

// Provide a definition for the HEIR StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the HEIR dialect.
// def HEIR_StructType :
//     DialectType<HEIR_Dialect, CPred<"$_self.isa<StructType>()">,
//                 "HEIR struct type">;

//===----------------------------------------------------------------------===//
// HEIR Operations
//===----------------------------------------------------------------------===//
def String : TypeDef<HEIR_Dialect, "String"> {

  let description = "A type for defining LUT function";

  let mnemonic = "string";

  let summary = "String";
}

def Plain : TypeDef<HEIR_Dialect, "Plain"> {

  let description = "A type for encoded Plaintext";

  let mnemonic = "plain";

  let summary = "Plain";

  // let parameters = (ins
  //     "Type":$plaintextType    
  // );
}

def Int : TypeDef<HEIR_Dialect, "Int"> {

  let description = "A type for Integer";

  let mnemonic = "int";

  let summary = "int";
}

def Float : TypeDef<HEIR_Dialect, "Float"> {

  let description = "A type for Float";

  let mnemonic = "float";

  let summary = "float";
}

def F64Vector : TypeDef<HEIR_Dialect, "F64Vector"> {

  let description = "A type for vector of float";

  let mnemonic = "f64vector";

  let summary = "f64vector";
}

def PlainVector : TypeDef<HEIR_Dialect, "PlainVector"> {

  let description = "A type for vector of plain";

  let mnemonic = "plainvector";

  let summary = "plainvector";
}

def LWECipher : TypeDef<HEIR_Dialect, "LWECipher"> {

  let description = "A type for LWE Cipher";

  let mnemonic = "lwecipher";

  let summary = "lwecipher";

  let parameters = (ins "Type":$plaintextType);

  let assemblyFormat = "`<` $plaintextType `>`";
}

def RLWECipher : TypeDef<HEIR_Dialect, "RLWECipher"> {

  let description = "A type for RLWE Cipher";

  let mnemonic = "rlwecipher";

  let summary = "rlwecipher";

  let parameters = (ins
      "Type":$plaintextType,
      "int":$size
  );

  let assemblyFormat = "`<` $size `x` $plaintextType `>`";
}

def RGSWCipher : TypeDef<HEIR_Dialect, "RGSWCipher"> {

  let description = "A type for RGSW Cipher";

  let mnemonic = "rgswcipher";

  let summary = "rgswcipher";
}

def LWECipherVector : TypeDef<HEIR_Dialect, "LWECipherVector"> {

  let description = "A type for vector of LWE Cipher";

  let mnemonic = "lweciphervec";

  let summary = "lweciphervec";

  let parameters = (ins
      "Type":$plaintextType,
      "int":$size
  );

  let assemblyFormat = "`<` $size`x`$plaintextType `>`";
}

def LWECipherMatrix: TypeDef<HEIR_Dialect, "LWECipherMatrix"> {

  let description = "A type for matrix of LWE Cipher";

  let mnemonic = "lweciphermat";

  let summary = "lweciphermat";

  let parameters = (ins
      "Type":$plaintextType,
      "int":$row,
      "int":$column
  );

  let assemblyFormat = "`<` $row`x`$column`x`$plaintextType `>`";
}

def RLWECipherVector: TypeDef<HEIR_Dialect, "RLWECipherVector"> {

  let description = "A type for vector of RLWE Cipher, corresponding to a plaintext matrix";

  let mnemonic = "lweciphermat";

  let summary = "lweciphermat";

  let parameters = (ins
      "Type":$plaintextType,
      "int":$row,
      "int":$column
  );

  let assemblyFormat = "`<` $row`x`$column`x`$plaintextType `>`";
}

// Provide a definition of the types that are used within the HEIR dialect.
def HEIR_Type : AnyTypeOf<[String, Int, Float, F64Vector, Plain, PlainVector, 
                            LWECipher, RLWECipher, RGSWCipher, LWECipherVector,
                            LWECipherMatrix, RLWECipherVector]>;

//===----------------------------------------------------------------------===//
// HEIR Operations
//===----------------------------------------------------------------------===//

// We define a toy operation by inheriting from our base 'HEIR_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as '' as it is a pure operation
// and may be removed if dead.
def FHEDefineOp : HEIR_Op<"define",
    [NoSideEffect]> {
  let summary = "Define a new ciphertext";
  let description = [{
    Convert variable assignment in C program to ciphertext definition in FHE porgram
  }];

  // let arguments = (ins AnyType:$input);

  let results = (outs HEIR_Type:$output);
}

def FHEEncodeOp : HEIR_Op<"encode",
    [NoSideEffect]> {
  let summary = "LWE Encryption";
  let description = [{
    LWE Encryption fuction
  }];

  let arguments = (ins F32Attr:$message);
  let results = (outs AnyType:$cipher);

}

def FHEExtractOp : HEIR_Op<"extract_init", 
    [NoSideEffect]> {
  let summary = "Extract a single value from plain vector";
  let description = [{
    Extract a single value from plain vector.
  }];

  let arguments = (ins AnyType:$vector, Variadic<AnyType>:$i);
  let results = (outs AnyType:$result);
  // let assemblyFormat = "$vector `[` $i `]` attr-dict `:` type($result)";

}

def FHEExtractfinalOp : HEIR_Op<"extract", 
    [NoSideEffect]> {
  let summary = "Extract a single value from plain vector";
  let description = [{
    Extract a single value from plain vector.
  }];

  let arguments = (ins AnyTypeOf<[LWECipherVector,LWECipherMatrix,RLWECipher,RLWECipherVector]>:$vector,
                    AnyAttr:$col, OptionalAttr<AnyAttr>:$row);
  let results = (outs LWECipher:$result);
  // let assemblyFormat = [{
  //   $vector `[` $i `]`  attr-dict `:` type($result)
  // }];

}

def FHEInsertOp : HEIR_Op<"insert_init",
    []> {
  let summary = "Insert a single value to plain vector";
  let description = [{
    Insert a single value to plain vector.
  }];

  let arguments = (ins AnyType:$value, AnyType:$memref, Variadic<AnyType>:$index);
  // let results = (outs AnyType:$result);
}


def FHEInsertfinalOp : HEIR_Op<"insert",
    []> {
  let summary = "Insert a single value to plain vector";
  let description = [{
    Insert a single value to plain vector.
  }];

  let arguments = (ins AnyType:$value, AnyType:$memref,  AnyAttr:$col, OptionalAttr<AnyAttr>:$row);
  // let results = (outs AnyType:$result);
}

def LWEMulOp : HEIR_Op<"lwemul",
    [NoSideEffect]> {
  let summary = "LWECipher multiplication operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  // let arguments = (ins LWECipher:$lhs, LWECipher:$rhs);
  let arguments = (ins
    // AnyType:$lhs, AnyType:$rhs
    Variadic<AnyType>:$x
  );
  let results = (outs 
    // Variadic<AnyType>:$output);
    AnyType:$output);
}

def RLWEMulOp : HEIR_Op<"rlwemul",
    [NoSideEffect]> {
  let summary = "LWECipher multiplication operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins
    Variadic<AnyType>:$x
  );
  let results = (outs 
    AnyType:$output);
}


def LWEAddOp : HEIR_Op<"lweadd",
    [NoSideEffect]> {
  let summary = "LWECipher addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  // let arguments = (ins LWECipher:$lhs, LWECipher:$rhs);
  let arguments = (ins
    // AnyType:$lhs, AnyType:$rhs
    Variadic<AnyType>:$x
  );
  let results = (outs 
    // Variadic<AnyType>:$output);
    AnyType:$output);

}

def LWESubOp : HEIR_Op<"lwesub",
    [NoSideEffect]> {
  let summary = "LWECipher addition operation";
  let description = [{
    The "sub" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  // let arguments = (ins LWECipher:$lhs, LWECipher:$rhs);
  let arguments = (ins
    // AnyType:$lhs, AnyType:$rhs
    Variadic<AnyType>:$x
  );
  let results = (outs 
    // Variadic<AnyType>:$output);
    AnyType:$output);

}

def FHERotateOp: HEIR_Op<"rotate", [NoSideEffect]> {
  let summary = "Rotate the RLWECipher in slot form.";

  let arguments = (ins 
    AnyTypeOf<[RLWECipher, LWECipherVector]>:$cipher,
    SI32Attr:$i
  );

  let results = (outs 
    AnyTypeOf<[RLWECipher, LWECipherVector]>:$output
  );

  let hasFolder = 1;
}

def FHEMaterializeOp: HEIR_Op<"materialize", [NoSideEffect]> {
  let summary = "No-op operation used to preserve consistency of type system during type conversion";

  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let hasFolder = 1;
}

def FHECmpOp: HEIR_Op<"compare", []> {
  let summary = "LWE comparing operation";

  let arguments = (ins Arith_CmpFPredicateAttr:$predicate,
                       AnyType:$lhs,
                       AnyType:$rhs);
  
  let results = (outs AnyType:$result);

  // let hasFolder = 1;
}

def FHESelectOp: HEIR_Op<"select", []> {
  let summary = "Replace arith.select with this operation";

  let arguments = (ins AnyType:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);
  
  let results = (outs AnyType:$result);
}

def FHERepackOp: HEIR_Op<"repack", []> {
  let summary = "Repack LWECiphers to a RLWECipher";

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);
}

def FHEVectorLoadOp: HEIR_Op<"vector_load_init", []> {
  let summary = "Replace polygeist.subindex to load a vector/slice from a matrix";

  let arguments = (ins AnyType:$memref, AnyType:$indices);

  let results = (outs AnyType:$result);

  // let hasFolder = 1;

}

def FHEVectorLoadfinalOp: HEIR_Op<"vector_load", []> {
  let summary = "Same as FHEVectorOp, but replace Value Type indices with Attr Type";

  let arguments = (ins AnyType:$memref, AnyAttr:$index);
  
  let results = (outs AnyType:$result);
}

def FHELUTForAddOp: HEIR_Op<"lut_half", []> {
  let summary = "LUT Operation for AddOp in Min Value";

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);
}

def FHELUTForSubOp: HEIR_Op<"lut_abshalf", []> {
  let summary = "LUT Operation for AddOp in Min Value";

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);
}

def FHELUTForLTOp: HEIR_Op<"lut_lt", []> {
  let summary = "LUT Opeartion for < 0";

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);
} 

def FHELUTForGTOp: HEIR_Op<"lut_gt", []> {
  let summary = "LUT Opeartion for > 0";

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);
} 

def FHELUTOp: HEIR_Op<"lut", []> {
  let summary = "LUT Opeartion without function evaluation";

  let arguments = (ins AnyType:$input);

  let results = (outs AnyType:$result);
} 

def FHEFuncCallOp: HEIR_Op<"call", []> {
  let summary = "Function call to replace Func::CallOp";

  let description = [{
    To replace Func::CallOp, since type conflict between function call 
    and function definition is occured in FuncToHEIR pass.
  }];
  
  let arguments = (ins
    Arg<StrAttr, "the homomorphic function to call">:$callee,
    Arg<OptionalAttr<ArrayAttr>, "the order of operands and further attributes">:$args,
    Arg<OptionalAttr<ArrayAttr>, "template arguments">:$template_args,
    Variadic<AnyType>:$operands
  );

  let results = (outs AnyType);
}

#endif // HEIR_OPS