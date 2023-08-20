add_heir_dialect_library(HEIRHEIRDialect
        HEIRDialect.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/heir

        DEPENDS
        MLIRHEIRIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )
