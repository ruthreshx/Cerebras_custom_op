import torch, pytest
import custom_module

@pytest.mark.parametrize("input", [
    (torch.tensor([3.0, 0.0, -2.0])),   
    (torch.tensor([5.0])), 
    (torch.tensor([1.5, -1.5])),  
    (torch.tensor([0.0])), 
    (torch.tensor([float('inf'), -float('inf'), 1.0, 0.0])), # +/- Nan and +/- Inf check
])
def test_entr(input):

    # Create an instance of Heaviside
    ch = custom_module.Entr()

    # Perform custom Heaviside
    custom_entr = ch.compute_entr(input)

    torch_entr = torch.special.entr(input)

    # Assert that the custom result matches the expected output
    assert torch.allclose(custom_entr, torch_entr, rtol=1e-05, atol=1e-05), \
    f"custom_entr{input}) != torch.special.entr({input})"