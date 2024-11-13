#include <gtest/gtest.h>
#include "addmv.h"  // Include the header for your custom addmv class

// Define a struct for test parameters
struct AddMVTestParams {
    torch::Tensor input;
    torch::Tensor matrix;
    torch::Tensor vector;
    double beta;
    double alpha;
};

// Define the CustomaddmvTest class
class CustomAddMVTest : public ::testing::TestWithParam<AddMVTestParams> {
protected:
    custom_namespace::AddMV* custom_addmv;

    // Setup function to initialize the test fixture
    void SetUp() override {
        custom_addmv = new custom_namespace::AddMV();
    }

    // Tear down function to clean up after tests
    void TearDown() override {
        delete custom_addmv;
    }
};

// Instantiate parameterized tests with different input parameters
INSTANTIATE_TEST_SUITE_P(
    CustomAddMVTests,
    CustomAddMVTest,
    ::testing::Values(
                // Basic valid case
        AddMVTestParams{torch::tensor({1.0, 2.0, 3.0}), torch::rand({3, 2}), torch::rand({2}), 1.0, 1.0, },
        
        // Zero input tensor
        AddMVTestParams{torch::tensor({0.0, 0.0, 0.0}), torch::rand({3, 2}), torch::rand({2}), 0.0, 1.0, },

        // Negative values in input
        AddMVTestParams{torch::tensor({-1.0, 2.0, 3.0}), torch::rand({3, 2}), torch::rand({2}), 1.0, 1.0, },

        // Different scaling factors
        AddMVTestParams{torch::tensor({1.0, 2.0, 3.0}), torch::rand({3, 3}), torch::rand({3}), 2.0, 1.0, },

        // Matrices with small values
        AddMVTestParams{torch::tensor({0.5, 0.5}), torch::rand({2, 2}), torch::rand({2}), 1.0, 2.0, },

        // Single element tensor
        AddMVTestParams{torch::tensor({1.0}), torch::rand({1, 2}), torch::rand({2}), 1.0, 1.0, },

        // Empty tensor case
        AddMVTestParams{torch::tensor({}), torch::rand({0, 2}), torch::rand({2}), 1.0, 1.0, },

        // Random tensor with uniform distribution
        AddMVTestParams{torch::rand({100}), torch::rand({100, 50}), torch::rand({50}), 1.0, 1.0, },

        // Random tensor with normal distribution 
        AddMVTestParams{torch::randn({10}), torch::randn({10, 50}), torch::randn({50}), 1.0, 1.0}

    )
);

// Parameterized test for Custom AddMV function
TEST_P(CustomAddMVTest, ParameterizedAddMV) {
    AddMVTestParams params = GetParam();

    torch::Tensor result = custom_addmv->compute_addmv(params.input, params.matrix, params.vector, params.beta, params.alpha);
    
    // Expected result: manually perform the matrix multiplication and addition
    torch::Tensor expected = torch::addmv(params.input, params.matrix, params.vector, params.beta, params.alpha);

    // Compare results with a tolerance of 1e-5
    ASSERT_TRUE(torch::allclose(result, expected, 1e-5));
}