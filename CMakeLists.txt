cmake_minimum_required(VERSION 2.8)

project(OmniFeature)

set(CMAKE_CXX_STANDARD 11)

# required libraries
find_package(OpenCV REQUIRED)

include_directories(
   ${PROJECT_SOURCE_DIR}
   ${PROJECT_SOURCE_DIR}/include
   ${OpenCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} SHARED
    src/CamModelGeneral.cpp
    src/Campare.cpp 
    src/Frame.cpp
    src/ORBextractor.cpp
    src/system.cpp
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS} 
)

##for fangshan dataset
add_executable(Cube_Omni_Feature 
   ${PROJECT_SOURCE_DIR}/Examples/main_cube.cpp
)
target_link_libraries(Cube_Omni_Feature ${PROJECT_NAME})

##for lafida dataset
add_executable(Lafida_Omni_Feature 
   ${PROJECT_SOURCE_DIR}/Examples/main_lafida.cpp
)
target_link_libraries(Lafida_Omni_Feature ${PROJECT_NAME})

##for simulation
add_executable(simulation
   ${PROJECT_SOURCE_DIR}/Examples/main_simulation.cpp
)
target_link_libraries(simulation ${PROJECT_NAME})