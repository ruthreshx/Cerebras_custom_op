import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 2, 2])),
        # (torch.Size([2, 2, 32, 84])),
        # (torch.Size([4, 3, 32, 64])),
    ),
)
def test_entr(input_shapes):

    # manual seed
    torch.manual_seed(4)

    # Generate Randn Input
    input = torch.randn(input_shapes)
    input = torch.clamp(input, -100, 100)

    # Create an instance of Entr
    ch = custom_module.Entr()

    # Perform custom Entr
    custom_entr = ch.compute_entr(input)

    torch_entr = torch.special.entr(input)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_entr, torch_entr, rtol=1e-02, atol=1e-02
    ), f"custom_entr{input}) != torch.special.entr({input})"
