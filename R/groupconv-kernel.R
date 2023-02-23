#' Base functionality of a group convolution kernel
#'
#' Kernel that maps between an incoming group and an outgoing group, both being semi-direct
#' products with R2.

#' @import torch
#' @export
GroupConvKernel <- torch::nn_module(
  "GroupConvKernel",
  #' @description define a group-to-group convolution kernel
  #' @param group the group to use
  #' @param kernel_size size of the convolution kernel
  #' @param in_channels number of channels in the input
  #' @param out_channels number of channels in the output layer
  initialize = function(group, kernel_size, in_channels, out_channels) {
    self$group <- group
    self$kernel_size <- kernel_size
    self$in_channels <- in_channels
    self$out_channels <- out_channels

    # Create spatial kernel grid. These are the coordinates on which our kernel weights are defined.
    self$register_buffer("grid_R2", torch::torch_stack(torch::torch_meshgrid(
      list(
        torch::torch_linspace(-1, 1, self$kernel_size),
        torch::torch_linspace(-1, 1, self$kernel_size)
      )
    )))

    # The kernel grid now also extends over the group H, as our input
    # feature maps contain an additional group dimension
    self$register_buffer("grid_H", self$group$elements())

    self$register_buffer("transformed_grid_R2xH", self$create_transformed_grid_R2xH())
  },
  #' @description Create a combined grid as the product of the transformed grids over R2 and H.
  create_transformed_grid_R2xH = function() {
    group_elements <- self$group$elements()
    # Transform the grid defined over R2 with the sampled group elements.
    transformed_grid_R2 <- self$group$left_action_on_R2(
      self$group$inverse(group_elements),
      self$grid_R2
    )

    # Transform the grid defined over H with the sampled group elements
    transformed_grid_H <- self$group$left_action_on_H(self$group$inverse(group_elements), self$grid_H)

    # Rescale values to between -1 and 1, we do this to please the torch grid_sample
    # function.
    transformed_grid_H <- self$group$normalize_group_elements(transformed_grid_H)

    # Create a combined grid as the product of the grids over R2 and H
    # repeat R2 along the group dimension, and repeat H along the spatial dimension
    # to create a [output_group_elem, num_group_elements, kernel_size, kernel_size, 3] grid
    transformed_grid <- torch::torch_cat(
      list
      (
        transformed_grid_R2$view(
          c(
            group_elements$numel(),
            1,
            self$kernel_size,
            self$kernel_size,
            2
          )
        )$`repeat`(c(1, group_elements$numel(), 1, 1, 1)),
        transformed_grid_H$view(
          c(
            group_elements$numel(),
            group_elements$numel(),
            1,
            1,
            1
          )
        )$`repeat`(c(1, 1, self$kernel_size, self$kernel_size, 1))
      ),
      dim = -1
    )
    transformed_grid
  },
  #' @description Sample convolution kernels for a given number of group elements
  #' @returns a filter bank extending over all input channels, containing kernels
  #' transformed for all output group elements.
  sample = function(sampled_group_elements) {
    stop("Not implemented.")
  }
)

#' A group convolution kernel that does interpolation
#'
#' @import torch
#' @export
InterpolatingGroupConvKernel <- torch::nn_module(
  "InterpolatingGroupConvKernel",
  inherit = GroupConvKernel,
  #' @description define a group-to-group convolution kernel
  #' @param group the group to use
  #' @param kernel_size size of the convolution kernel
  #' @param in_channels number of channels in the input
  #' @param out_channels number of channels in the output layer
  initialize = function(group, kernel_size, in_channels, out_channels) {
    super$initialize(group, kernel_size, in_channels, out_channels)

    # Create and initialize a set of weights, we will interpolate these
    # to create our transformed spatial kernels. Note that our weight
    # now also extends over the group H.
    self$weight <- torch::nn_parameter(
      torch::torch_zeros(
        c(self$out_channels, self$in_channels, self$group$elements()$numel(), self$kernel_size, self$kernel_size)
      )
    )
    # Initialize weights using kaiming uniform intialisation
    torch::nn_init_kaiming_uniform_(self$weight, a = sqrt(5))
  },
  #' @description Sample convolution kernels for a given number of group elements
  #' @returns a filter bank extending over all input channels, containing kernels
  #' transformed for all output group elements.
  sample = function() {
    # We fold the output channel dim into the input channel dim; this allows
    # us to use the torch grid_sample function.
    weight <- self$weight$view(
      c(
        1,
        self$out_channels * self$in_channels,
        self$group$elements()$numel(),
        self$kernel_size,
        self$kernel_size
      )
    )

    # We want a transformed set of weights for each group element so
    # we repeat the set of spatial weights along the output group axis
    weight <- weight$`repeat`(c(self$group$elements()$numel(), 1, 1, 1, 1))

    # Sample the transformed kernels
    transformed_weight <- torch::nnf_grid_sample(
      weight,
      self$transformed_grid_R2xH,
      mode = "bilinear",
      padding_mode = "zeros",
      align_corners = TRUE
    )

    # Separate input and output channels
    transformed_weight <- transformed_weight$view(
      c(
        self$group$elements()$numel(), # Output group elements (like in the lifting convolution)
        self$out_channels,
        self$in_channels,
        self$group$elements()$numel(), # Input group elements (due to the additional dimension of our feature map)
        self$kernel_size,
        self$kernel_size
      )
    )

    # Put the output channel dimension before the output group dimension.
    transformed_weight <- transformed_weight$transpose(1, 2)
    transformed_weight
  }
)
