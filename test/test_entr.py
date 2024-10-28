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
def test_entr(input_shapes):

    # Generate Randn Input
    input = torch.randn(input_shapes).bfloat16()

    # Create an instance of Heaviside
    ch = custom_module.Entr()

    # Perform custom Heaviside
    custom_entr = ch.compute_entr(input)

    torch_entr = torch.special.entr(input)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_entr, torch_entr, rtol=1e-05, atol=1e-05
    ), f"custom_entr{input}) != torch.special.entr({input})"
