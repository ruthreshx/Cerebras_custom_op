#include <pybind11/pybind11.h>
#include "minimum.h"
#include "heaviside.h"
#include "elu.h"
#include "entr.h"
#include "addmm.h"
#include "addmv.h"
#include "addbmm.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_module, m) {

    // Bind the Minimum class to Python
    py::class_<custom_namespace::Minimum>(m, "Minimum")
        .def(py::init<>())  // Bind the constructor
        .def("compute_min", &custom_namespace::Minimum::compute_min,  // Bind compute_min method

             py::arg("tensor1"), py::arg("tensor2"),  // Name the arguments
             "Compute the element-wise minimum between two tensors.\n\n"

             "Arguments:\n"
             "    tensor1 (torch.Tensor): The first tensor.\n"
             "    tensor2 (torch.Tensor): The second tensor.\n\n"

             "Returns:\n"
             "    torch.Tensor: A tensor containing the element-wise minimum.");

    // Bind the Heaviside class to Python
    py::class_<custom_namespace::Heaviside>(m, "Heaviside")
        .def(py::init<>())  // Bind constructor
        .def("compute_heaviside", &custom_namespace::Heaviside::compute_heaviside,  // Bind method

             py::arg("x"), py::arg("values"),  // Name the arguments

             "Compute the Heaviside step function.\n\n"

             "Arguments:\n"
             "    x (torch.Tensor): Input tensor for which to apply the Heaviside function.\n"
             "    values (torch.Tensor): Boundary condition tensor, returned where x == 0.\n\n"

             "Returns:\n"
             "    torch.Tensor: A tensor containing the result of the Heaviside step function.");

    // Bind the Elu class to Python
    py::class_<custom_namespace::ELU>(m, "ELU")
        .def(py::init<>())  // Bind the constructor
        .def("compute_elu", &custom_namespace::ELU::compute_elu,  // Bind the compute_elu method

             py::arg("tensor"), py::arg("alpha") = 1.0,  // Name the arguments and set default value for alpha

             "Compute the ELU activation function.\n\n"

             "Arguments:\n"
             "    tensor (torch.Tensor): The input tensor for which to compute ELU.\n"
             "    alpha (float): The alpha parameter controlling the slope for negative values (default: 1.0).\n\n"

             "Returns:\n"
             "    torch.Tensor: A tensor containing the ELU-activated values.");

    // Bind the Entr class to Python
    py::class_<custom_namespace::Entr>(m, "Entr")
        .def(py::init<>())  // Bind constructor
        .def("compute_entr", &custom_namespace::Entr::compute_entr,  // Bind method

             py::arg("tensor"),  // Name the argument

             "Compute the Entr function.\n\n"

             "Arguments:\n"
             "    tensor (torch.Tensor): The input tensor for which to compute the Entr function.\n\n"

             "Returns:\n"
             "    torch.Tensor: A tensor containing the result of applying the Entr function.");

    // Bind the Addmm class to Python
    py::class_<custom_namespace::AddMM>(m, "AddMM")
        .def(py::init<>())  // Bind constructor
        .def("compute_addmm", &custom_namespace::AddMM::compute_addmm,  // Bind method

            py::arg("input"), 
            py::arg("mat1"), 
            py::arg("mat2"), 
            py::kw_only(), 
            py::arg("beta") = 1.0, 
            py::arg("alpha") = 1.0,

            "Custom addmm operation with PyTorch.\n\n"

            "Arguments:\n"
            "    input (torch.Tensor): The tensor to which the result will be added.\n"
            "    mat1 (torch.Tensor): The first matrix for multiplication.\n"
            "    mat2 (torch.Tensor): The second matrix for multiplication.\n"
            "    beta (float, optional): Scalar multiplier for `input`. Default is 1.0.\n"
            "    alpha (float, optional): Scalar multiplier for `mat1 @ mat2`. Default is 1.0.\n\n"

            "Returns:\n"
            "    torch.Tensor: The result of the addmm operation.");

    // Bind the Addmv class to Python
    py::class_<custom_namespace::AddMV>(m, "AddMV")
        .def(py::init<>())  // Bind Constructor
        .def("compute_addmv", &custom_namespace::AddMV::compute_addmv,  // Method

        py::arg("input"), 
        py::arg("matrix"),
        py::arg("vector"),
        py::kw_only(), 
        py::arg("beta") = 1.0,
        py::arg("alpha") = 1.0,
            
        "Custom addmv performs matrix-vector multiplication and adds a bias vector.\n\n"

        "Arguments:\n"
        "   input (torch.Tensor): A 1D tensor representing the bias to be added.\n"
        "   matrix (torch.Tensor): A 2D tensor representing the matrix.\n"
        "   vector (torch.Tensor): A 1D tensor representing the vector.\n"
        "   alpha (float, optional): Scalar multiplier for matrix-vector product. Default is 1.0.\n"
        "   beta (float, optional): Scalar multiplier for the bias vector. Default is 1.0.\n\n"

        "Returns:\n"
        "    torch.Tensor: The result of the addmv operation.");

    // Bind the Addbmm class to Python
    py::class_<custom_namespace::AddBMM>(m, "AddBMM")
        .def(py::init<>())  // Constructor
        .def("compute_addbmm", &custom_namespace::AddBMM::compute_addbmm,  // Method
            
        py::arg("input"), 
        py::arg("batch1"), 
        py::arg("batch2"),
        py::kw_only(), 
        py::arg("alpha") = 1.0, 
        py::arg("beta") = 1.0,
        
        "Custom addbmm Performs batched matrix-matrix multiplication and adds a bias tensor.\n\n"

        "Arguments:\n"
        "   input (torch.Tensor): The tensor to which the result will be added.\n"
        "   batch1 (torch.Tensor): A batch of matrices (3D tensor) for multiplication.\n\n"
        "   batch2 (torch.Tensor): A second batch of matrices (3D tensor) for multiplication.\n"
        "   alpha (float, optional): Scalar multiplier for the batch matrix-matrix product. Default is 1.0.\n"
        "   beta (float, optional): Scalar multiplier for the input tensor. Default is 1.0.\n"

        "Returns:\n"
        "    torch.Tensor: The result of the addbmm operation.");
    

}
