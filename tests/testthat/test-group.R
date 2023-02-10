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
