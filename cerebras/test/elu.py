import torch, pytest
import custom_module

@pytest.mark.parametrize("input, alpha", [
    (torch.tensor([3.0, 0.0, -2.0]), 1.0),   
    (torch.tensor([5.0]), -3.0), 
    (torch.tensor([1.5, -1.5]), -5.5),  
    (torch.tensor([0.0]), 2.5),  
])
def test_elu(input, alpha):

    # Create an instance of Heaviside
    ch = custom_module.ELU()

    # Perform custom Heaviside
    custom_elu = ch.compute_elu(input, alpha)

    torch_elu = torch.nn.functional.elu(input, alpha)

    # Assert that the custom result matches the expected output
    assert torch.allclose(custom_elu, torch_elu, rtol=1e-05, atol=1e-05), \
    f"custom_elu({input}, {alpha}) != torch.elu({input}, {alpha})"