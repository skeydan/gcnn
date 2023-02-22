#' Lifting convolution
#'
#' A convolution module taking as input a feature map defined over R2, and returning
#' a feature map over the semi-direct product of R2 and the symmetry group employed.
#' @import torch
#' @export
LiftingConvolution <- torch::nn_module(
  "LiftingConvolution",
  #' @description implement a lifting convolution module
  #' @param group the group to use in lifting
  #' @param kernel_size size of the convolution kernel
  #' @param in_channels number of channels in the input
  #' @param out_channels number of channels in the output layer
  initialize = function(group, kernel_size, in_channels, out_channels) {
    self$kernel <- InterpolatingLiftingKernel(
      group = group,
      kernel_size = kernel_size,
      in_channels = in_channels,
      out_channels = out_channels
    )
  },
  # input of size [batch_dim, in_channels, spatial_dim_1, spatial_dim_2]
  # returns a function on a homogeneous space of the group, of size
  # [batch_dim, out_channels, num_group_elements, spatial_dim_1, spatial_dim_2]
  forward = function(x) {
    # obtain convolution kernels transformed under the group
    conv_kernels <- self$kernel$sample()

    # apply lifting convolution, note that the reshape folds the group dimension of the kernel into the output channel dimension.
    x <- nnf_conv2d(
      input = x,
      weight = conv_kernels$reshape(
        c(
          self$kernel$out_channels * self$kernel$group$elements()$numel(),
          self$kernel$in_channels,
          self$kernel$kernel_size,
          self$kernel$kernel_size
        )
      ),
    )

    # reshape [batch_dim, in_channels * num_group_elements, spatial_dim_1, spatial_dim_2]
    # into [batch_dim, in_channels, num_group_elements, spatial_dim_1, spatial_dim_2],
    # separating channel and group dimensions.
    x <- x$view(c(
      -1,
      self$kernel$out_channels,
      self$kernel$group$elements()$numel(),
      x$shape[4],
      x$shape[3]
    ))

    x
  }
)
