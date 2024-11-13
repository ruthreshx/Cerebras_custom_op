import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input, matrix, vector",
    (
        (torch.randn([2]), torch.randn([2, 3]), torch.randn([3])),
        (torch.randn([4]), torch.randn([4, 5]), torch.randn([5])),
        (torch.ones(3), torch.zeros(3, 2), torch.ones(2)),  # Zero tensor and ones
        (torch.ones(2), torch.ones(2, 3), torch.ones(3)),  # Ones tensor
    ),
)
@pytest.mark.parametrize("alpha", [-55.0, -33.0, 0.0, 11.0, 22.0])
@pytest.mark.parametrize("beta", [-88.0, -44.0, 0.0, 22.0, 11.0])
def test_AddMV(input, matrix, vector, alpha, beta):

    # Create an instance of AddMV
    ch = custom_module.AddMV()

    # Perform custom AddMV
    custom_AddMV = ch.compute_addmv(input, matrix, vector, alpha=alpha, beta=beta)

    torch_AddMV = torch.addmv(input, matrix, vector, alpha=alpha, beta=beta)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_AddMV, torch_AddMV, rtol=1e-03, atol=1e-03
    ), f"custom_AddMV({input}, {matrix}, {vector}) != torch_AddMV({input}, {matrix}, {vector})"
