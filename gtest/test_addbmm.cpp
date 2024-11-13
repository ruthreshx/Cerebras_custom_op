#include <gtest/gtest.h>
#include "addbmm.h"  // Include the header for your custom addbmm class

// Define a struct for test parameters
struct AddBMMTestParams {
    torch::Tensor input;
    torch::Tensor batch1;
    torch::Tensor batch2;
    double beta;
    double alpha;
};

// Define the CustomAddBMMTest class
class CustomAddBMMTest : public ::testing::TestWithParam<AddBMMTestParams> {
protected:
    custom_namespace::AddBMM* custom_addbmm;

    // Setup function to initialize the test fixture
    void SetUp() override {
        custom_addbmm = new custom_namespace::AddBMM();
    }

    // Tear down function to clean up after tests
    void TearDown() override {
        delete custom_addbmm;
    }
};

// Instantiate parameterized tests with different input parameters
INSTANTIATE_TEST_SUITE_P(
    CustomAddBMMTests,
    CustomAddBMMTest,
    ::testing::Values(

        // Random tensor with uniform distribution
        AddBMMTestParams{torch::rand({2, 3}), torch::rand({4, 2, 3}), torch::rand({4, 3, 3}), 1.0, 1.0},

        // Random tensor with uniform distribution
        AddBMMTestParams{torch::rand({2, 3}), torch::rand({4, 2, 8}), torch::rand({4, 8, 3}), 1.0, 1.0},

        // Random tensor with uniform distribution with larger batch size for mat1 and mat2 
        AddBMMTestParams{torch::rand({3, 6}), torch::rand({1, 3, 1}), torch::rand({1, 1, 6}), 2.0, 1.0}

    )
);

// Parameterized test for Custom AddBMM function
TEST_P(CustomAddBMMTest, ParameterizedAddBMM) {
    AddBMMTestParams params = GetParam();

    torch::Tensor result = custom_addbmm->compute_addbmm(params.input, params.batch1, params.batch2, params.beta, params.alpha);
    
    // Expected result: manually perform the matrix multiplication and addition
    torch::Tensor expected = torch::addbmm(params.input, params.batch1, params.batch2, params.beta, params.alpha);

    // Compare results with a tolerance of 1e2
    ASSERT_TRUE(torch::allclose(result, expected, 1e2));
}