test_that("LiftingConvolution", {
  lifting_conv <- LiftingConvolution(
    group = CyclicGroup(order = 4),
    kernel_size = 5,
    in_channels = 3,
    out_channels = 8
  )

  x <- torch::torch_randn(c(2, 3, 32, 32))
  expect_equal(lifting_conv(x)$shape[1:3], c(2, 8, 4))
})

