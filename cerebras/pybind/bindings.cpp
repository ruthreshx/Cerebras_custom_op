#include <pybind11/pybind11.h>
#include "minimum.h"
#include "heaviside.h"
#include "elu.h"
#include "entr.h"

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

}
