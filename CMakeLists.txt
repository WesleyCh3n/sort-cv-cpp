cmake_minimum_required(VERSION 3.18)

project(sort-cpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)

option(BUILD_EXAMPLES "Build examples" ON)

#==============================================================================
# Third Party Dependencies
#==============================================================================
find_package(OpenCV REQUIRED)

#==============================================================================
# Build Libraries
#==============================================================================
include(GNUInstallDirs)
add_library(track
  src/hungarian.cpp
)
target_link_libraries(track PUBLIC
  ${OpenCV_LIBS}
)
target_include_directories(track PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
target_include_directories(track PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
set_target_properties(track PROPERTIES PUBLIC_HEADER "src/track.hpp")

include(CMakePackageConfigHelpers)
list(APPEND INSTALL_TARGETS
  track
)
set(CONFIG_EXPORT_NAME track-config)
set(CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/track)
set(PROJECT_CONFIG ${CMAKE_CURRENT_BINARY_DIR}/${CONFIG_EXPORT_NAME}.cmake)

install(TARGETS ${INSTALL_TARGETS} EXPORT ${CONFIG_EXPORT_NAME}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/track)
export(TARGETS ${INSTALL_TARGETS}
       NAMESPACE track::
       FILE ${PROJECT_CONFIG})
install(EXPORT ${CONFIG_EXPORT_NAME}
        DESTINATION "${CMAKE_DIR}"
        NAMESPACE track::
)

#==============================================================================
# Build Examples
#==============================================================================
if(BUILD_EXAMPLES)
  add_executable(main
    ${CMAKE_CURRENT_SOURCE_DIR}/example/main.cpp
  )
  target_link_libraries(main PRIVATE track)
endif()
