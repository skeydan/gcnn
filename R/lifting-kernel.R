#' Base functionality of a lifting kernel
#'
#' The kernel lifts a signal from the input domain to the semi-direct product of the input domain
#' and the intended symmetry group.

#' @import torch
#' @export
LiftingKernel <- torch::nn_module(
  "LiftingKernel",
  #' @description define a lifting kernel
  #' @param group the group to use in lifting
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
      )))$to(self$group$identity.device))

    # Transform the grid by the elements in this group.
    self$register_buffer("transformed_grid_R2", self$create_transformed_grid_R2())
  },
  #' @description Transform the created grid by the group action of each group element.
  #' This yields a grid (over H) of spatial grids (over R2). In other words,
  #' a list of grids, each index of which is the original spatial grid transformed by
  #' a corresponding group element in H.
  create_transformed_grid_R2 = function() {
    group_elements <- self$group$elements()
    # Transform the grid defined over R2 with the sampled group elements.
    transformed_grid = self$group$left_action_on_R2(
      self$group$inverse(group_elements),
      self$grid_R2
    )
    transformed_grid
  },
  #' @description Sample convolution kernels for a given number of group elements
  #' @param sampled_group_elements the group elements over which to sample the convolution kernels
  #' @returns a filter bank extending over all input channels, containing kernels
  #' transformed for all output group elements.
  sample = function() {
    stop("Not implemented.")
  }
)

