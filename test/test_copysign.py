import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input, other",
    (
        (torch.randn([2, 3]), torch.randn([2, 3])),
        (torch.randn([4, 4]), torch.randn([4, 4])),
        (torch.ones(2, 3), torch.ones(2, 3)),  # Ones tensor
        (torch.randn(3, 3, 4, 4), torch.randn(3, 3, 4, 4)),  # 4D
        (torch.randn(4, 8, 16, 16), torch.randn(4, 8, 16, 16)),  # 4D Bigger shapes
        (
            torch.tensor([3.0, -3.34, -4.32, 2.0]),
            torch.tensor([-10.0, 33.34, -53.322, 1.0]),
        ),  # Negative values
        (
            torch.tensor([0.0, -11.34, -56.32, -190.0]),
            torch.tensor([-0.0, -33.34, 90.322, -32.0]),
        ),  # Negative values
    ),
)
def test_copySign(input, other):

    # Create an instance of CopySign
    ch = custom_module.CopySign()

    # Perform custom CopySign
    custom_copysign = ch.compute_copysign(input, other)

    torch_copysign = torch.copysign(input, other)

    print("\n custom_copysign ===> ", custom_copysign)
    print("torch_copysign  ===> ", torch_copysign)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_copysign, torch_copysign, rtol=1e-03, atol=1e-03
    ), f"custom_copysign({input}, {other}) != torch_copysign({input}, {other})"
