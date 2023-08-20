add_heir_conversion_library(HEIRLWEToRLWE
        LWEToRLWE.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HEIRHEIRDialect
        )
