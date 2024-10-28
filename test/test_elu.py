import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("alpha", [-1.0, 0.0, 1.0, 3.0])
def test_elu(input_shapes, alpha):

    # Generate Randn Input
    input = torch.randn(input_shapes).bfloat16()

    # Create an instance of Heaviside
    ch = custom_module.ELU()

    # Perform custom Heaviside
    custom_elu = ch.compute_elu(input, alpha)

    torch_elu = torch.nn.functional.elu(input, alpha)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_elu, torch_elu, rtol=1e-02, atol=1e-02
    ), f"custom_elu({input}, {alpha}) != torch.elu({input}, {alpha})"
