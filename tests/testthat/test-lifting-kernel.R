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

  # Values of shifted kernels, by element and position in the kernel. This has to be 2d since the kernel is.
  expect_equal(as.numeric(transformed_grid_R2[1, 1, 1, ]), c(-1, -1))
  expect_equal(as.numeric(transformed_grid_R2[2, 1, 1, ]), c(1, -1))

  # For visualization, fold both spatial dimensions of the kernel (not R2!) into a single dimension
  transformed_grid_R2 <- transformed_grid_R2$reshape(
    c(transformed_grid_R2$shape[1],
      transformed_grid_R2$shape[2] * transformed_grid_R2$shape[3],
      2)
  )
  # then can create, for each group element, a scatter plot of values in the two R2 dimensions
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


