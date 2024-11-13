// test_entr.cpp
#include <gtest/gtest.h>
#include <torch/torch.h>
#include "entr.h"

namespace custom_namespace {

    // Test for positive inputs to compute_entr
    TEST(EntrTest, PositiveInput) {
        custom_namespace::Entr entr_instance;
        torch::Tensor input = torch::tensor({2.0, 3.0, 4.0});
        
        // Call compute_entr
        torch::Tensor result = entr_instance.compute_entr(input);
        
        // Expected output for -x * log(x) for each element in {0.5, 1.0, 2.0}
        torch::Tensor expected = torch::special::entr(input);

        // Test if the result is close to the expected output
        ASSERT_TRUE(torch::allclose(result, expected, 1e-2));
    }

    // Test for zero input to compute_entr
    TEST(EntrTest, ZeroInput) {
        custom_namespace::Entr entr_instance;
        torch::Tensor input = torch::tensor({0.0});
        
        // Call compute_entr
        torch::Tensor result = entr_instance.compute_entr(input);

        // Expected output: zero since - * log(-0) = 0 by convention
        torch::Tensor expected = torch::special::entr(input);

        // Test if the result matches the expected output
        ASSERT_TRUE(torch::allclose(result, expected, 1e-2));
    }

    // Test for negative inputs to compute_entr
    TEST(EntrTest, NegativeInput) {
        custom_namespace::Entr entr_instance;
        torch::Tensor input = torch::tensor({-1.5, -2.5});
        
        // Call compute_entr
        torch::Tensor result = entr_instance.compute_entr(input);

        // Expected output: zero for each element, as the function is defined as zero for non-positive values
        torch::Tensor expected = torch::special::entr(input);

        // Test if the result matches the expected output
        ASSERT_TRUE(torch::allclose(result, expected, 1e-2));
    }

} // namespace custom_namespace
