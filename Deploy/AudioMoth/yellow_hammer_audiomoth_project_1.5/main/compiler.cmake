set(CMAKE_SYSTEM_NAME Generic)
#set(CMAKE_SYSTEM_PROCESSOR arm)

set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(LINKER arm-none-eabi-ld)

# adapted from CMakeCXXInformation.cmake
# original rule uses ${CMAKE_CXX_COMPILER} instead
set(CMAKE_CXX_LINK_EXECUTABLE
    "${LINKER} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
)

# adapted from CMakeCInformation.cmake
# original rule uses ${CMAKE_C_COMPILER} instead
set(CMAKE_C_LINK_EXECUTABLE
    "${LINKER} <FLAGS> <CMAKE_CXX_LINK_FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> <LINK_LIBRARIES>"
)

