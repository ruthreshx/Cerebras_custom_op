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
def test_heaviside(input_shapes):

    # Generate Randn Input
    input = torch.randn(input_shapes).bfloat16()
    values = torch.randn(input_shapes).bfloat16()

    # Create an instance of Heaviside
    ch = custom_module.Heaviside()

    # Perform custom Heaviside
    custom_heaviside = ch.compute_heaviside(input, values)

    torch_heaviside = torch.heaviside(input, values)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_heaviside, torch_heaviside, rtol=1e-05, atol=1e-05
    ), f"custom_heaviside({input}, {values}) != torch.heaviside({input}, {values})"
