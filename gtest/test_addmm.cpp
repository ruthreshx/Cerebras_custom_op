#include <gtest/gtest.h>
#include "addmm.h"  // Include the header for your custom addmm class

// Define a struct for test parameters
struct AddMMTestParams {
    torch::Tensor input;
    torch::Tensor mat1;
    torch::Tensor mat2;
    double beta;
    double alpha;
};

// Define the CustomAddMMTest class
class CustomAddMMTest : public ::testing::TestWithParam<AddMMTestParams> {
protected:
    custom_namespace::AddMM* custom_addmm;

    // Setup function to initialize the test fixture
    void SetUp() override {
        custom_addmm = new custom_namespace::AddMM();
    }

    // Tear down function to clean up after tests
    void TearDown() override {
        delete custom_addmm;
    }
};

// Instantiate parameterized tests with different input parameters
INSTANTIATE_TEST_SUITE_P(
    CustomAddMMTests,
    CustomAddMMTest,
    ::testing::Values(
                // Basic valid case
        AddMMTestParams{torch::tensor({1.0, 2.0, 3.0}), torch::rand({3, 2}), torch::rand({2, 3}), 1.0, 1.0, },
        
        // Zero input tensor
        AddMMTestParams{torch::tensor({0.0, 0.0, 0.0}), torch::rand({3, 2}), torch::rand({2, 3}), 0.0, 1.0, },

        // Negative values in input
        AddMMTestParams{torch::tensor({1.0, 2.0, 4.0}), torch::rand({3, 2}), torch::rand({2, 3}), 1.0, 1.0, },

        // Different scaling factors
        AddMMTestParams{torch::tensor({1.0, 2.0, 3.0}), torch::rand({3, 2}), torch::rand({2, 3}), 2.0, 1.0, },

        // Matrices with small values
        AddMMTestParams{torch::tensor({0.5, 0.5}), torch::rand({2, 2}), torch::rand({2, 2}), 1.0, 2.0, },

        // Single element tensor
        AddMMTestParams{torch::tensor({1.0}), torch::rand({1, 2}), torch::rand({2, 1}), 1.0, 1.0, },

        // Empty tensor case
        AddMMTestParams{torch::tensor({}), torch::rand({0, 2}), torch::rand({2, 0}), 1.0, 1.0, },

        // Random tensor with uniform distribution
        AddMMTestParams{torch::rand({10, 10}), torch::rand({10, 50}), torch::rand({50, 10}), 1.0, 1.0, },

        // Random tensor with normal distribution 
        AddMMTestParams{torch::randn({10, 10}), torch::randn({10, 50}), torch::randn({50, 10}), 1.0, 1.0}

    )
);

// Parameterized test for Custom AddMM function
TEST_P(CustomAddMMTest, ParameterizedAddMM) {
    AddMMTestParams params = GetParam();

    torch::Tensor result = custom_addmm->compute_addmm(params.input, params.mat1, params.mat2, params.beta, params.alpha);
    
    // Expected result: manually perform the matrix multiplication and addition
    torch::Tensor expected = torch::addmm(params.input, params.mat1, params.mat2, params.beta, params.alpha);

    // Compare results with a tolerance of 1e-3
    ASSERT_TRUE(torch::allclose(result, expected, 1e-3));
}