# Assuming you're in a sub-directory "tests" in your project directory

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Create the testJsons directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/testJsons)

# Copy test.json to the testJsons directory
configure_file(${CMAKE_SOURCE_DIR}/tests/test.json ${CMAKE_CURRENT_BINARY_DIR}/testJsons/test.json COPYONLY)

include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/madrona
)

# Add your test file(s) to the add_executable command
add_executable(my_tests
    bicyclemodel.cpp
    test_utils.cpp
    CollisionDetectionTests.cpp
    observationTest.cpp
    EgocentricRoadObservationTests.cpp
)

# Link against required libraries. Ensure that the paths and names are correct.
target_link_libraries(my_tests
    gtest_main
    gpudrive_mgr
    nlohmann_json::nlohmann_json
    madrona_libcxx
)

# Include GoogleTest module functions and discover tests in the executable
include(GoogleTest)
gtest_discover_tests(my_tests)
