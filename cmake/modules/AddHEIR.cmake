# Authors: HECO
# Modified by Zian Zhao
# Copyright:
# Copyright (c) 2020 ETH Zurich.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include_guard()

function(add_heir_dialect dialect dialect_namespace)
  add_mlir_dialect(${ARGV})
  add_dependencies(heir-headers MLIR${dialect}IncGen)
endfunction()


function(add_heir_library name)
  add_mlir_library(${ARGV})
  add_heir_library_install(${name})
endfunction()

# Adds a HEIR library target for installation.  This should normally only be
# called from add_heir_library().
function(add_heir_library_install name)
  install(TARGETS ${name} COMPONENT ${name} EXPORT HEIRTargets)
  set_property(GLOBAL APPEND PROPERTY HEIR_ALL_LIBS ${name})
  set_property(GLOBAL APPEND PROPERTY HEIR_EXPORTS ${name})
endfunction()

function(add_heir_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY HEIR_DIALECT_LIBS ${name})
  add_heir_library(${ARGV} DEPENDS heir-headers)
endfunction()

function(add_heir_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY HEIR_CONVERSION_LIBS ${name})
  add_heir_library(${ARGV} DEPENDS heir-headers)
endfunction()
