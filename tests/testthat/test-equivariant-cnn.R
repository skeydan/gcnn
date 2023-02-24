test_that("GroupEquivariantCNN", {
  cnn <- GroupEquivariantCNN(
    group = CyclicGroup(order = 4),
    kernel_size = 5,
    in_channels = 1,
    out_channels = 1,
    num_hidden = 2,
    hidden_channels = 16
  )
  img <- torch::torch_randn(c(4, 1, 32, 32))

  expect_equal(cnn(img)$shape, c(4, 1))
})

# test_that("GroupEquivariantCNN on MNIST", {
#   dir <- "/tmp/mnist"
#
#   train_ds <- torchvision::mnist_dataset(
#     dir,
#     download = TRUE,
#     transform = torchvision::transform_to_tensor
#   )
#
#   test_ds <- torchvision::mnist_dataset(
#     dir,
#     train = FALSE,
#     transform = function(x) {
#       x |>
#         torchvision::transform_to_tensor() |>
#         torchvision::transform_random_rotation(
#           degrees = c(0, 360),
#           resample = 2,
#           fill = 0
#         )
#     }
#   )
#
#   train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
#   test_dl <- dataloader(test_ds, batch_size = 128)
#
#   fitted <- GroupEquivariantCNN |>
#     luz::setup(
#       loss = torch::nn_cross_entropy_loss(),
#       optimizer = torch::optim_adam,
#       metrics = list(
#         luz::luz_metric_accuracy()
#       )
#     ) |>
#     luz::set_hparams(
#       group = CyclicGroup(order = 4),
#       kernel_size = 5,
#       in_channels = 1,
#       out_channels = 10,
#       num_hidden = 4,
#       hidden_channels = 16
#     ) |>
#     luz::set_opt_hparams(lr = 1e-2, weight_decay = 1e-4) |>
#     luz::fit(train_dl, epochs = 10, valid_data = test_dl) # valid acc: 0.87
#
#   default_cnn <- nn_module(
#     "default_cnn",
#     initialize = function(kernel_size, in_channels, out_channels, num_hidden, hidden_channels) {
#       self$conv1 <- torch::nn_conv2d(in_channels, hidden_channels, kernel_size)
#       self$convs <- torch::nn_module_list()
#       for (i in 1:num_hidden) {
#         self$convs$append(torch::nn_conv2d(hidden_channels, hidden_channels, kernel_size))
#       }
#       self$avg_pool <- torch::nn_adaptive_avg_pool2d(1)
#       self$final_linear <- torch::nn_linear(hidden_channels, out_channels)
#     },
#     forward = function(x) {
#       x <- x |>
#         self$conv1() |>
#         (\(.) torch::nnf_layer_norm(., .$shape[2:4]))() |>
#         torch::nnf_relu()
#       for (i in 1:(length(self$convs))) {
#         x <- x |>
#           (\(.) self$convs[[i]](.))() |>
#           (\(.) torch::nnf_layer_norm(., .$shape[2:4]))() |>
#           torch::nnf_relu()
#       }
#       x <- x |>
#         self$avg_pool() |>
#         torch::torch_squeeze() |>
#         self$final_linear()
#       x
#     }
#   )
#   fitted <- default_cnn |>
#     luz::setup(
#       loss = torch::nn_cross_entropy_loss(),
#       optimizer = torch::optim_adam,
#       metrics = list(
#         luz::luz_metric_accuracy()
#       )
#     ) |>
#     luz::set_hparams(
#       kernel_size = 5,
#       in_channels = 1,
#       out_channels = 10,
#       num_hidden = 4,
#       hidden_channels = 32
#     ) %>%
#     luz::set_opt_hparams(lr = 1e-2, weight_decay = 1e-4) |>
#     luz::fit(train_dl, epochs = 10, valid_data = test_dl) # valid acc: 0.42
# })
