import torch, pytest
import custom_module


@pytest.mark.parametrize(
    "input, mat1, mat2",
    (
        (torch.randn([2, 2]), torch.randn([2, 3]), torch.randn([3, 2])),
        (torch.randn([4, 5]), torch.randn([4, 6]), torch.randn([6, 5])),
        (torch.zeros(3, 2), torch.ones(3, 4), torch.ones(4, 2)),  # Zero tensor and ones
        (torch.ones(2, 3), torch.ones(2, 4), torch.ones(4, 3)),  # Ones tensor
    ),
)
@pytest.mark.parametrize("alpha", [-10.0, -5.0, 0.0, 5.0, 10.0])
@pytest.mark.parametrize("beta", [-20.0, -10.0, 0.0, 10.0, 20.0])
def test_addmm(input, mat1, mat2, alpha, beta):

    # Create an instance of Addmm
    ch = custom_module.AddMM()

    # Perform custom Addmm
    custom_addmm = ch.compute_addmm(input, mat1, mat2, alpha=alpha, beta=beta)

    torch_addmm = torch.addmm(input, mat1, mat2, alpha=alpha, beta=beta)

    # Assert that the custom result matches the expected output
    assert torch.allclose(
        custom_addmm, torch_addmm, rtol=1e-03, atol=1e-03
    ), f"custom_addmm({input}, {mat1}, {mat2}) != torch_addmm({input}, {mat1}, {mat2})"
