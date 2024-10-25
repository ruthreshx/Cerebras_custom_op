import torch, pytest
import custom_module

@pytest.mark.parametrize("input, other", [
    (torch.tensor([3.0, 0.0, -2.0]), torch.tensor([3.0, 0.0, 3.0])),   
    (torch.tensor([5.0]), torch.tensor([-3.0])), 
    (torch.tensor([1.5, -1.5]), torch.tensor([3.0, 2.0])),  
    (torch.tensor([0.0]), torch.tensor([0.0])),  
])
def test_minimum(input, other):

    # Create an instance of Minimum
    ch = custom_module.Minimum()

    # Perform custom Minimum
    custom_minimum = ch.compute_min(input, other)

    torch_minimum = torch.minimum(input, other)


    # Assert that the custom result matches the expected output
    assert torch.allclose(custom_minimum, torch_minimum, rtol=1e-05, atol=1e-05), \
    f"custom_minimum({input}, {other}) != torch.minimum({input}, {other})"