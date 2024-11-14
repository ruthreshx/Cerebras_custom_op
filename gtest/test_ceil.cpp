#include <gtest/gtest.h>
#include "ceil.h"  // Include the header for your custom Ceil class

// Define a struct for test parameters
struct CeilTestParams {
    torch::Tensor input;

};

// Define the CustomCeilTest class
class CustomCeilTest : public ::testing::TestWithParam<CeilTestParams> {
protected:
    custom_namespace::Ceil* custom_ceil;

    // Setup function to initialize the test fixture
    void SetUp() override {
        custom_ceil = new custom_namespace::Ceil();
    }

    // Tear down function to clean up after tests
    void TearDown() override {
        delete custom_ceil;
    }
};

// Instantiate parameterized tests with different input parameters
INSTANTIATE_TEST_SUITE_P(
    CustomCeilTests,
    CustomCeilTest,
    ::testing::Values(
                
        // Basic valid case
        CeilTestParams{torch::tensor({1.0, 2.0, 3.0, 4.0})},
        
        // Zero input tensor
        CeilTestParams{torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0})},

        // Negative values in input
        CeilTestParams{torch::tensor({-1.0, -2.0, -4.0})},

        // Bigger inputs 2D
        CeilTestParams{torch::rand({32, 128})},

        // Bigger inputs 4D
        CeilTestParams{torch::rand({3, 11, 22, 33})},

        // Single element tensor
        CeilTestParams{torch::tensor({1.0})},

        // Empty tensor case
        CeilTestParams{torch::rand({0})}


    )
);

// Parameterized test for Custom Ceil function
TEST_P(CustomCeilTest, ParameterizedCeil) {
    CeilTestParams params = GetParam();

    torch::Tensor result = custom_ceil->compute_ceil(params.input);
    
    // Expected result
    torch::Tensor expected = torch::ceil(params.input);

    // Compare results with a tolerance of 1e-5
    ASSERT_TRUE(torch::allclose(result, expected, 1e-5));
}