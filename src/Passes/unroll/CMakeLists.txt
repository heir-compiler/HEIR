add_heir_conversion_library(HEIRUnrollLoop
        UnrollLoop.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HEIRHEIRDialect
        )