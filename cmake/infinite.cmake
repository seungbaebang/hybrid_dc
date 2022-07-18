set(INFINITE_ROOT "${CMAKE_CURRENT_LIST_DIR}/..")
set(INFINITE_INCLUDE_DIR "${INFINITE_ROOT}/fmm" "${INFINITE_ROOT}/curve" "${INFINITE_ROOT}/bem" "${INFINITE_ROOT}/biem" "${INFINITE_ROOT}/util")

file(GLOB INFINITE_SRCFILES ${INFINITE_ROOT}/fmm/*.cpp ${INFINITE_ROOT}/fmm/*.h 
                            ${INFINITE_ROOT}/curve/*.cpp ${INFINITE_ROOT}/curve/*.h
                            ${INFINITE_ROOT}/bem/*.cpp ${INFINITE_ROOT}/bem/*.h
                            ${INFINITE_ROOT}/biem/*.cpp ${INFINITE_ROOT}/biem/*.h
                            ${INFINITE_ROOT}/util/*.cpp ${INFINITE_ROOT}/util/*.h)

option(INFINITE_USE_STATIC_LIBRARY "Use infinite as static library" ON)
option(INFINITE_USE_OPENMP "Use OpenMP" ON)

# libigl
option(LIBIGL_USE_PREBUILT_LIBRARIES "Use prebuilt libraries"       ON)
option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" ON)
# option(LIBIGL_WITH_CGAL              "Use CGAL"                     OFF)
# option(LIBIGL_WITH_COMISO            "Use CoMiso"                   OFF)
# option(LIBIGL_WITH_CORK              "Use Cork"                     OFF)
# option(LIBIGL_WITH_EMBREE            "Use Embree"                   OFF)
# option(LIBIGL_WITH_MATLAB            "Use Matlab"                   OFF)
# option(LIBIGL_WITH_MOSEK             "Use MOSEK"                    OFF)
# option(LIBIGL_WITH_OPENGL            "Use OpenGL"                   ON)
# option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"                     ON)
# option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use ImGui"                    OFF)
option(LIBIGL_WITH_PNG               "Use PNG"                      ON)
# option(LIBIGL_WITH_TETGEN            "Use Tetgen"                   OFF)
# option(LIBIGL_WITH_TRIANGLE          "Use Triangle"                 OFF)
# option(LIBIGL_WITH_PREDICATES        "Use exact predicates"         OFF)
# option(LIBIGL_WITH_XML               "Use XML"                      OFF)

if(LIBIGL_USE_PREBUILT_LIBRARIES)
    find_package(LIBIGL REQUIRED)
    find_package (Eigen3 3.3 REQUIRED NO_MODULE)
else()
    include(libigl)
endif()

if(INFINITE_USE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()

if(INFINITE_USE_STATIC_LIBRARY)
  #message(STATUS "inf include dir: ${INFINITE_INCLUDE_DIR}")
  # file(GLOB INFINITE_SRCFILES "${INFINITE_INCLUDE_DIR}/*.cpp")
  file(GLOB INFINITE_SRCFILES "${INFINITE_ROOT}/fmm/*.cpp"
                              "${INFINITE_ROOT}/curve/*.cpp"
                              "${INFINITE_ROOT}/bem/*.cpp"
                              "${INFINITE_ROOT}/biem/*.cpp"
                              "${INFINITE_ROOT}/util/*.cpp")

  add_library(infinite STATIC ${INFINITE_SRCFILES})
else()
  add_library(infinite INTERFACE)
endif()

include_directories(${INFINITE_INCLUDE_DIR})

if(INFINITE_USE_OPENMP)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pthread")
endif()

if(INFINITE_USE_STATIC_LIBRARY)
  target_include_directories(infinite PUBLIC ${INFINITE_INCLUDE_DIR})
  target_link_libraries(infinite PUBLIC igl::core igl::png Eigen3::Eigen)
  target_compile_definitions(infinite PUBLIC -DINFINITE_STATIC_LIBRARY)
else()
  target_include_directories(infinite INTERFACE ${INFINITE_INCLUDE_DIR})
  target_link_libraries(infinite INTERFACE igl::core igl::png Eigen3::Eigen)
endif()

if(OpenMP_CXX_FOUND)

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        execute_process(
            COMMAND brew --prefix libomp 
            RESULT_VARIABLE BREW_OMP
            OUTPUT_VARIABLE BREW_OMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )

        include_directories(${BREW_OMP_PREFIX}/include)

        if(INFINITE_USE_STATIC_LIBRARY)
            target_link_libraries(infinite PUBLIC OpenMP::OpenMP_CXX)
            target_compile_definitions(infinite PUBLIC -DINFINITE_USE_OPENMP)
        else()
            target_link_libraries(infinite INTERFACE OpenMP::OpenMP_CXX)
            target_compile_definitions(infinite INTERFACE -DINFINITE_USE_OPENMP)
        endif()

    elseif()
        include_directories(${OpenMP_CXX_INCLUDE_DIRS})

        if(INFINITE_USE_STATIC_LIBRARY)
            target_link_libraries(infinite PUBLIC OpenMP::OpenMP_CXX)
            target_compile_definitions(infinite PUBLIC -DINFINITE_USE_OPENMP)
        else()
            target_link_libraries(infinite INTERFACE OpenMP::OpenMP_CXX)
            target_compile_definitions(infinite INTERFACE -DINFINITE_USE_OPENMP)
        endif()
    endif()
    
endif()
