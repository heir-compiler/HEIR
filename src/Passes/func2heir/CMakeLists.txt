add_heir_conversion_library(HEIRFuncToHEIR
        FuncToHEIR.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HEIRHEIRDialect
        )

