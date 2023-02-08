test_that("CyclicGroup", {
  c4 <- CyclicGroup(order = 4)

  elems <- c4$elements()
  e <- elems[1]
  g1 <- elems[2]
  g2 <- elems[3]
  g3 <- elems[4]

  prod <- c4$product(g1, e)
  expect_equal(as.numeric(prod), pi/2, tolerance = 1e-7)
  prod <- c4$product(g1, g2)
  expect_equal(as.numeric(prod), as.numeric(g3), tolerance = 1e-7)

  inv <- c4$inverse(g3)
  expect_equal(as.numeric(c4$product(inv, g3)), as.numeric(e))

  e_rep <- c4$matrix_representation(e)
  g1_rep <- c4$matrix_representation(g1)
  g2_rep <- c4$matrix_representation(g2)
  mm <- torch::torch_matmul(g1_rep$t(), g2_rep)
  expect_true(torch::torch_allclose(g1_rep, mm, atol = 1e-6))
})
