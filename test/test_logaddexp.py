import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input, other",
    (
        (torch.randn([2, 3]), torch.randn([2, 3])),
        (torch.randn([8, 5]), torch.randn([8, 5])),
        (torch.ones(7, 2), torch.zeros(7, 2)),  # Zero tensor and ones
        (torch.ones(9, 3), torch.ones(9, 3)),  # Ones tensor
        (torch.randn(6, 6, 4, 4), torch.randn(6, 6, 4, 4)),  # 4D
        (torch.randn(6, 8, 32, 64), torch.randn(6, 8, 32, 64)),  # 4D Bigger shapes
    ),
)
def test_logaddexp(input, other):

    # Create an instance of LogAddExp
    ch = custom_module.LogAddExp()

    # Perform custom LogAddExp
    custom_logaddexp = ch.compute_logaddexp(input, other)

    torch_logaddexp = torch.logaddexp(input, other)
    
    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_logaddexp, torch_logaddexp, rtol=1e-05, atol=1e-05
    ), f"custom_logaddexp({input}, {other} != torch_logaddexp({input}, {other})"
