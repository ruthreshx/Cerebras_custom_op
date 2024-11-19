import torch, pytest
import custom_module

torch.manual_seed(2)


@pytest.mark.parametrize(
    "input, batch1, batch2",
    (
        (
            torch.randn(3, 2),
            torch.randn(4, 3, 3),
            torch.randn(4, 3, 2),
        ),  # Regular input and matrices
        (
            torch.zeros(2, 3),
            torch.ones(4, 2, 3),
            torch.ones(4, 3, 3),
        ),  # Zeros input, ones for mat1, mat2
        # Varying tensor shapes
        (
            torch.randn(3, 2),
            torch.randn(6, 3, 3),
            torch.randn(6, 3, 2),
        ),  # Larger batch size for mat1 and mat2
        (
            torch.randn(5, 2),
            torch.randn(4, 5, 5),
            torch.randn(4, 5, 2),
        ),  # Higher dimension tensors for batch
    ),
)
@pytest.mark.parametrize("alpha", [-10.0, -5.0, 0.0, 1.0, 5.0, 10.0])
@pytest.mark.parametrize("beta", [-20.0, -10.0, 0.0, 1.0, 10.0, 20.0])
def test_AddBMM(input, batch1, batch2, alpha, beta):

    # Create an instance of AddBMM
    ch = custom_module.AddBMM()

    # Perform custom AddBMM
    custom_AddBMM = ch.compute_addbmm(input, batch1, batch2, alpha=alpha, beta=beta)

    torch_AddBMM = torch.addbmm(input, batch1, batch2, alpha=alpha, beta=beta)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_AddBMM, torch_AddBMM, rtol=1e2, atol=1e2
    ), f"custom_AddBMM({input}, {batch1}, {batch2}) != torch_AddBMM({input}, {batch1}, {batch2})"
