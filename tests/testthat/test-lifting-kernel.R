test_that("LiftingKernel", {
  lifting_kernel <- LiftingKernel(
    group = CyclicGroup(order = 4),
    kernel_size = 7,
    in_channels = 1,
    out_channels = 1
  )

  transformed_grid_R2 <- lifting_kernel$transformed_grid_R2

  # The grid has a shape of [num_group_elements, kernel_size, kernel_size, dim_fmap_domain(R^2)]
  expect_equal(transformed_grid_R2$shape, c(4, 7, 7, 2))

  # for plotting, see https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb

})


test_that("InterpolatingLiftingKernel", {
  lifting_kernel <- InterpolatingLiftingKernel(
    group = CyclicGroup(order = 4),
    kernel_size = 5,
    in_channels = 2,
    out_channels = 1
  )
  weights <- lifting_kernel$sample()
  expect_equal(weights$shape, c(1, 4, 2, 5, 5))

})


