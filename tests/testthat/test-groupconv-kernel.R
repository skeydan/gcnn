test_that("GroupConvKernel", {
  groupconv_kernel <- GroupConvKernel(
    group = CyclicGroup(order = 4),
    kernel_size = 7,
    in_channels = 1,
    out_channels = 1
  )

  # Sample the group
  group_elements <- groupconv_kernel$group$elements()

  # Transform the grid defined over R2 with the sampled group elements
  transformed_grid_R2 <- groupconv_kernel$group$left_action_on_R2(
    groupconv_kernel$group$inverse(group_elements),
    groupconv_kernel$grid_R2
  )

 })
