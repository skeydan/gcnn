test_that("GroupConvKernel", {
  groupconv_kernel <- GroupConvKernel(
    group = CyclicGroup(order = 4),
    kernel_size = 7,
    in_channels = 1,
    out_channels = 1
  )

  # Sample the group
  group_elements <- groupconv_kernel$group$elements()

  # check the action on R2
  # for plotting, see https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb
  transformed_grid_R2 <- groupconv_kernel$group$left_action_on_R2(
    groupconv_kernel$group$inverse(group_elements),
    groupconv_kernel$grid_R2
  )
  expect_equal(transformed_grid_R2$shape, c(4, 7, 7, 2))

  # check action on group
  # for plotting, see https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb
  transformed_grid_H <- groupconv_kernel$group$left_action_on_H(
    groupconv_kernel$group$inverse(group_elements), groupconv_kernel$grid_H
  )
  expect_equal(as.numeric(transformed_grid_H$trace()), 0)

  # check overall action
  # for plotting, see https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb
  transformed_grid_R2xH <- groupconv_kernel$transformed_grid_R2xH
  expect_equal(transformed_grid_R2xH$shape, c(4, 4, 7, 7, 3))
 })

test_that("InterpolatingGroupConvKernel", {
  groupconv_kernel <- InterpolatingGroupConvKernel(
    group = CyclicGroup(order = 4),
    kernel_size = 5,
    in_channels = 2,
    out_channels = 8
  )
  weights <- groupconv_kernel$sample()
  expect_equal(weights$shape, c(8, 4, 2, 4, 5, 5))

  # for plotting, see https://github.com/phlippe/uvadlc_notebooks/blob/master/docs/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.ipynb

})


