#' Group equivariant CNN
#'
#' @import torch
#' @export
GroupEquivariantCNN <- torch::nn_module(
  "GroupConvolution",
  #' @description implement a lifting convolution module
  #' @param group the group to use in lifting
  #' @param kernel_size size of the convolution kernel
  #' @param in_channels number of channels in the input
  #' @param out_channels number of channels in the output layer
  #' @param num_hidden number of hidden layers
  #' @param hidden_channels number of channels in the hidden layers
  initialize = function(group, kernel_size, in_channels, out_channels, num_hidden, hidden_channels) {
    # Create the lifing convolution.
    self$lifting_conv <- LiftingConvolution(
      group = group,
      in_channels = in_channels,
      out_channels = hidden_channels,
      kernel_size = kernel_size
    )

    # Create a set of group convolutions.
    self$gconvs <- torch::nn_module_list()
    for (i in 1:num_hidden) {
      self$gconvs$append(
        GroupConvolution(
          group = group,
          in_channels = hidden_channels,
          out_channels = hidden_channels,
          kernel_size = kernel_size
        )
      )
    }

    # And a final linear layer for classification.
    self$final_linear <- torch::nn_linear(hidden_channels, out_channels)
  },
  forward = function(x) {
    # Lift and disentangle features in the input.
    x <- x |>
      self$lifting_conv() |>
      (\(.) torch::nnf_layer_norm(., .$shape[2:5]))() |>
      torch::nnf_relu()

    # Apply group convolutions.
    for (i in 1:(length(self$gconvs))) {
      x <- x |>
        (\(.) self$gconvs[[i]](.))() |>
        (\(.) torch::nnf_layer_norm(., .$shape[2:5]))() |>
        torch::nnf_relu()
    }

    # to ensure equivariance, apply max pooling over group and spatial dims
    x <- x |>
      (\(.) torch::torch_mean(., dim = c(3, 4, 5)))()
    x <- self$final_linear(x$squeeze())
    x
  }
)
