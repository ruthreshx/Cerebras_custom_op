#include <gtest/gtest.h>
#include "logaddexp.h"  // Include the header for your custom LogAddExp class

// Define a struct for test parameters
struct LogAddExpTestParams {
    torch::Tensor input;
    torch::Tensor other;

};

// Define the CustomLogAddExpTest class
class CustomLogAddExpTest : public ::testing::TestWithParam<LogAddExpTestParams> {
protected:
    custom_namespace::LogAddExp* custom_logaddexp;

    // Setup function to initialize the test fixture
    void SetUp() override {
        custom_logaddexp = new custom_namespace::LogAddExp();
    }

    // Tear down function to clean up after tests
    void TearDown() override {
        delete custom_logaddexp;
    }
};

// Instantiate parameterized tests with different input parameters
INSTANTIATE_TEST_SUITE_P(
    CustomLogAddExpTests,
    CustomLogAddExpTest,
    ::testing::Values(
                
        // Basic valid case
        LogAddExpTestParams{torch::tensor({1.0, 2.0, 3.0, 4.0}), torch::rand({4})},
        
        // Zero input tensor
        LogAddExpTestParams{torch::tensor({0.0, 0.0, 0.0, 0.0, 0.0}), torch::rand({5})},

        // Negative values in input
        LogAddExpTestParams{torch::tensor({-1.0, -2.0, -4.0}), torch::rand({3})},

        // Bigger inputs 2D
        LogAddExpTestParams{torch::rand({64, 128}), torch::rand({64, 128})},

        // Bigger inputs 4D
        LogAddExpTestParams{torch::rand({64, 128, 32, 32}), torch::rand({64, 128, 32, 32})},

        // Single element tensor
        LogAddExpTestParams{torch::tensor({1.0}), torch::rand({1})},

        // Empty tensor case
        LogAddExpTestParams{torch::rand({0, 2}), torch::rand({2, 0})}


    )
);

// Parameterized test for Custom LogAddExp function
TEST_P(CustomLogAddExpTest, ParameterizedLogAddExp) {
    LogAddExpTestParams params = GetParam();

    torch::Tensor result = custom_logaddexp->compute_logaddexp(params.input, params.other);
    
    // Expected result
    torch::Tensor expected = torch::logaddexp(params.input, params.other);

    // Compare results with a tolerance of 1e-5
    ASSERT_TRUE(torch::allclose(result, expected, 1e-5));
}