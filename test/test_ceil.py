import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input",
    (
        (torch.randn([3])),
        (torch.randn([7, 5])),
        (torch.zeros(5, 2)),  # Zero tensor and ones
        (torch.ones(2)),  # Ones tensor
        (torch.tensor([-2, -0.0, 0.0, 1.0, 2.0])),  # Ones tensor
        (torch.randn([4, 5, 4, 5])),  # 4D
        (torch.randn([32, 16, 32, 18])),  # Bigger shape
    ),
)
def test_ceil(input):

    # Create an instance of Ceil
    ch = custom_module.Ceil()

    # Perform custom Ceil
    custom_ceil = ch.compute_ceil(input)

    torch_ceil = torch.ceil(input)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_ceil, torch_ceil, rtol=1e-05, atol=1e-05
    ), f"custom_ceil({input}) != torch_ceil({input})"
