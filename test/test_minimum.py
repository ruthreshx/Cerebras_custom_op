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
def test_minimum(input_shapes):

    # Generate Randn Input
    input = torch.randn(input_shapes).bfloat16()
    other = torch.randn(input_shapes).bfloat16()

    # Create an instance of Minimum
    ch = custom_module.Minimum()

    # Perform custom Minimum
    custom_minimum = ch.compute_min(input, other)

    torch_minimum = torch.minimum(input, other)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_minimum, torch_minimum, rtol=1e-05, atol=1e-05
    ), f"custom_minimum({input}, {other}) != torch.minimum({input}, {other})"
