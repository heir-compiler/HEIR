add_heir_conversion_library(HEIRUnfold
        Unfold.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HEIRHEIRDialect
        )