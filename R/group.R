#' An abstract group.

#' @import torch
#' @export
Group <- torch::nn_module(
  "Group",
  #' @description define a group
  #' @param dimension dimension of the group
  #' @param identity the identity element of the group
  initialize = function(dimension, identity) {
    self$dimension <- dimension
    self$register_buffer("identity", torch::torch_tensor(identity))
  },
  #' @description obtain a tensor containing all group elements in this group
  elements = function() {
    stop("Not implemented.")
  },
  #' @description define the group product on two elements
  product = function(h, h_prime) {
    stop("Not implemented.")
  },
  #' @description obtain inverse for a group element
  inverse = function(h) {
    stop("Not implemented.")
  },
  #' @description group action of an element from the subgroup H on a vector in R2.
  #'   For efficiency we implement this batchwise.
  left_action_on_R2 = function(h_batch, x_batch) {
    stop("Not implemented.")
  },
  #' @description group action of elements of H on other elements in H itself. Comes down to group product.
  #'   For efficiency we implement this batchwise. Each element in `h_batch` is applied to each element
  #'   in `h_prime_batch`.
  left_action_on_H = function(h_batch, h_prime_batch) {
    stop("Not implemented.")
  },
  #' @description obtain a matrix representation in R2 for an element `h`
  matrix_representation = function(h) {
    stop("Not implemented.")
  },
  #' @description calculate the determinant of the representation of a group element `h`
  determinant = function(h) {
    stop("Not implemented.")
  },
  #' @description  map the group elements to an interval `[-1, 1]`. We use this to create
  #'   a standardized input for obtaining weights over the group.
  normalize_group_elements = function(h) {
    stop("Not implemented.")
  }
)

#' A cyclic group of order `n`.

#' @import torch
#' @export
CyclicGroup <- torch::nn_module(
  "CyclicGroup",
  inherit = Group,
  #' @description define a group
  #' @param order the order of the group
  initialize = function(order) {
    super$initialize(dimension = 1, identity = 0)
    self$order <- order
  },
  #' @description obtain a tensor containing all group elements in this group
  elements = function() {
    torch_linspace(
      start = 0,
      end = 2 * pi * (self$order - 1) / (self$order),
      steps = self$order,
      device = self$identity$device
    )
  },
  #' @description define the group product on two elements
  product = function(h, h_prime) {
    torch::torch_remainder(h + h_prime, 2 * pi)
  },
  #' @description obtain inverse for a group element
  inverse = function(h) {
    torch::torch_remainder(-h, 2 * pi)
  },
  #' @description group action of an element from the subgroup H on a vector in R2.
  #'   For efficiency we implement this batchwise.
  left_action_on_R2 = function(h_batch, x_batch) {
    # Create a tensor containing representations of each of the group
    # elements in the input. Creates a tensor of size [batch_size, 2, 2].
    h_reps <- torch::torch_zeros(dim(h_batch)[1], 2, 2)
    for (i in 1:(dim(h_batch)[1])) h_reps[i, , ] <- self$matrix_representation(h_batch[i])

    # Transform the r2 input grid with each representation to end up with a transformed
    # grid of dimensionality [num_group_elements, spatial_dim_y, spatial_dim_x, 2].
    out <- torch::torch_einsum("boi,ixy->bxyo", list(h_reps, x_batch))

    # Afterwards (because grid_sample assumes our grid is y,x instead of x,y)
    # we swap x and y coordinate values with a roll along final dimension.
    out$roll(shifts = 1, dims = -1)
  },
  #' @description group action of elements of H on other elements in H itself. Comes down to group product.
  #'   For efficiency we implement this batchwise. Each element in `h_batch` is applied to each element
  #'   in `h_prime_batch`.
  left_action_on_H = function(h_batch, h_prime_batch) {
    # The elements in h_batch work on the elements in h_prime_batch directly, through
    # the group product. Each element in h_batch is applied to each element in h_prime_batch.
    transformed_h_batch <- self$product(
      h_batch$`repeat`(h_prime_batch$dim()[1]),
      h_prime_batch$unsqueeze(-1)
    )
    transformed_h_batch
  },
  #' @description obtain a matrix representation in R2 for an element `h`
  matrix_representation = function(h) {
    cos_t <- torch_cos(h)
    sin_t <- torch_sin(h)
    torch::torch_stack(list(cos_t, -sin_t, sin_t, cos_t))$view(c(2,2))
  },
  #' @description  map the group elements to an interval `[-1, 1]`. We use this to create
  #'   a standardized input for obtaining weights over the group.
  normalize_group_elements = function(h) {
    largest_elem <- 2 * pi * (self$order - 1) / self$order
    2 * h / largest_elem - 1
  }
)



