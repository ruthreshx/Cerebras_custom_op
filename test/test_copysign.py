import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input, other",
    (
        (torch.randn([2]), torch.randn([2, 3])),
        (torch.randn([4]), torch.randn([4, 5])),
        (torch.ones(3), torch.zeros(3, 2)),  # Zero tensor and ones
        (torch.ones(2), torch.ones(2, 3)),  # Ones tensor
        (torch.randn(3, 3, 4, 4), torch.randn(3, 3, 4, 4)),  # 4D
        (torch.randn(4, 8, 16, 16), torch.randn(4, 8, 16, 16)),  # 4D Bigger shapes
    ),
)
def test_copySign(input, other):

    # Create an instance of CopySign
    ch = custom_module.CopySign()

    # Perform custom CopySign
    custom_copysign = ch.compute_copysign(input, other)

    torch_copysign = torch.copysign(input, other)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_copysign, torch_copysign, rtol=1e-03, atol=1e-03
    ), f"custom_copysign({input}, {other}) != torch_copysign({input}, {other})"
