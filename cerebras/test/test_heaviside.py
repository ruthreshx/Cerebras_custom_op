import torch, pytest
import custom_module

@pytest.mark.parametrize("input, values", [
    (torch.tensor([3.0, 0.0, -2.0]), torch.tensor([3.0, 0.0, 3.0])),   
    (torch.tensor([5.0]), torch.tensor([-3.0])), 
    (torch.tensor([1.5, -1.5]), torch.tensor([3.0, 2.0])),  
    (torch.tensor([0.0]), torch.tensor([0.0])),  
])
def test_heaviside(input, values):

    # Create an instance of Heaviside
    ch = custom_module.Heaviside()

    # Perform custom Heaviside
    custom_heaviside = ch.compute_heaviside(input, values)

    torch_heaviside = torch.heaviside(input, values)

    # Assert that the custom result matches the expected output
    assert torch.allclose(custom_heaviside, torch_heaviside, rtol=1e-05, atol=1e-05), \
    f"custom_heaviside({input}, {values}) != torch.heaviside({input}, {values})"