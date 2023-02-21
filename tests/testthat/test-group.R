test_that("CyclicGroup", {
  c4 <- CyclicGroup(order = 4)

  elems <- c4$elements()
  e <- elems[1]
  g1 <- elems[2]
  g2 <- elems[3]
  g3 <- elems[4]

  prod <- c4$product(g1, e)
  expect_equal(as.numeric(prod), pi / 2, tolerance = 1e-7)
  prod <- c4$product(g1, g2)
  expect_equal(as.numeric(prod), as.numeric(g3), tolerance = 1e-7)

  inv <- c4$inverse(g3)
  expect_equal(as.numeric(c4$product(inv, g3)), as.numeric(e))

  e_rep <- c4$matrix_representation(e)
  g1_rep <- c4$matrix_representation(g1)
  g2_rep <- c4$matrix_representation(g2)
  mm <- torch::torch_matmul(g1_rep$t(), g2_rep)
  expect_true(torch::torch_allclose(g1_rep, mm, atol = 1e-6))

  h_batch <- torch::torch_stack(list(e, g1))
  h_prime_batch <- elems
  act <- c4$left_action_on_H(h_batch, h_prime_batch)
  expect_true(torch::torch_allclose(act[, 1], h_prime_batch))
  expect_true(torch::torch_allclose(torch_remainder(act[, 2] - act[, 1], 2 * pi), torch_zeros(4) + pi / 2))

  h_batch <- elems
  x_batch <- torch::torch_stack(
    torch::torch_meshgrid(
      list(
        torch::torch_linspace(-1, 1, 4),
        torch::torch_linspace(-1, 1, 4)
      )
    )
  )
  act <- c4$left_action_on_R2(h_batch, x_batch)
  expect_equal(act$shape, c(4, 4, 4, 2))
})

test_that("CyclicGroup rotates image as expected", {

  img_path <- system.file("imgs", "d.jpg", package = "gcnn")
  img <- torchvision::base_loader(img_path) |> torchvision::transform_to_tensor()
  # [2, 512, 512] since our image is 2-dimensional and has a width and height of 512 pixels
  img_grid_R2 <- torch::torch_stack(torch::torch_meshgrid(
    list(
      torch::torch_linspace(-1, 1, dim(img)[2]),
      torch::torch_linspace(-1, 1, dim(img)[3])
    )
  ))
  expect_equal(img_grid_R2$shape, c(2, 1024, 1024))

  c4 <- CyclicGroup(order = 4)
  elems <- c4$elements()
  g1 <- elems[2]
  g2 <- elems[3]
  g <- c4$product(g1, g2) # rotate clockwise by 90 degrees

  # Transform the image grid we just created with the matrix representation of
  # this group element. Note that we implemented this batchwise, so we add a dim.
  transformed_grid <- c4$left_action_on_R2(c4$inverse(g)$unsqueeze(1), img_grid_R2)
  expect_equal(transformed_grid$shape, c(1, 1024, 1024, 2))

  # This function samples an input tensor based on a grid using interpolation.
  # It is implemented for batchwise operations so we add a dimension to our input image.
  transformed_img <- torch::nnf_grid_sample(img$unsqueeze(1), transformed_grid, align_corners = TRUE, mode = "bilinear", padding_mode = "zeros")
  transformed_img <- transformed_img[1,..]$
    permute(c(2, 3, 1)) |> as.array()
  expect_equal(dim(transformed_img), c(1024, 1024, 3))
  #plot(as.raster(transformed_img))
})


