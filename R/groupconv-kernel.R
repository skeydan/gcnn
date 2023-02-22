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
