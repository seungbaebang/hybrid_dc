cmake_minimum_required(VERSION 3.1)
project(infinite_vector_graphics)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR/cmake}" ${CMAKE_PREFIX_PATH})

set(CMAKE_BUILD_TYPE Release)

#libigl
include(${PROJECT_SOURCE_DIR}/cmake/infinite.cmake)



if(INFINITE_USE_STATIC_LIBRARY)
    add_executable(bem_brute bem_brute.cpp)
else()
    add_executable(bem_brute ${INFINITE_SRCFILES} bem_brute.cpp)
endif()
target_link_libraries(bem_brute infinite)

if(INFINITE_USE_STATIC_LIBRARY)
    add_executable(hybrid_brute hybrid_brute.cpp)
else()
    add_executable(hybrid_brute ${INFINITE_SRCFILES} hybrid_brute.cpp)
endif()
target_link_libraries(hybrid_brute infinite)

if(INFINITE_USE_STATIC_LIBRARY)
    add_executable(hybrid_fmm hybrid_fmm.cpp)
else()
    add_executable(hybrid_fmm ${INFINITE_SRCFILES} hybrid_fmm.cpp)
endif()
target_link_libraries(hybrid_fmm infinite)

if(INFINITE_USE_STATIC_LIBRARY)
    add_executable(hybrid_fmm_adap hybrid_fmm_adap.cpp)
else()
    add_executable(hybrid_fmm_adap ${INFINITE_SRCFILES} hybrid_fmm_adap.cpp)
endif()
target_link_libraries(hybrid_fmm_adap infinite)

if(INFINITE_USE_STATIC_LIBRARY)
    add_executable(hybrid_fmm_zoom hybrid_fmm_zoom.cpp)
else()
    add_executable(hybrid_fmm_zoom ${INFINITE_SRCFILES} hybrid_fmm_zoom.cpp)
endif()
target_link_libraries(hybrid_fmm_zoom infinite)

if(INFINITE_USE_STATIC_LIBRARY)
    add_executable(bem_fmm_zoom bem_fmm_zoom.cpp)
else()
    add_executable(bem_fmm_zoom ${INFINITE_SRCFILES} bem_fmm_zoom.cpp)
endif()
target_link_libraries(bem_fmm_zoom infinite)
