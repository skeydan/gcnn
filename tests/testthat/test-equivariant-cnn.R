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

test_that("GroupEquivariantCNN on MNIST", {
  dir <- "/tmp/mnist"

  train_ds <- torchvision::mnist_dataset(
    dir,
    download = TRUE,
    transform = function(x) {
      x |>
        torchvision::transform_to_tensor() |>
        torchvision::transform_random_rotation(
          degrees = c(0, 360),
          resample = 2,
          fill = 0
        )
    }
  )

  test_ds <- torchvision::mnist_dataset(
    dir,
    train = FALSE,
    transform = torchvision::transform_to_tensor
  )

  train_dl <- dataloader(train_ds, batch_size = 128, shuffle = TRUE)
  test_dl <- dataloader(test_ds, batch_size = 128)

  geq_cnn <- GroupEquivariantCNN(
    group = CyclicGroup(order = 4),
    kernel_size = 5,
    in_channels = 1,
    out_channels = 10,
    num_hidden = 4,
    hidden_channels = 16
  )

  # tbd change pipe?
  fitted <- GroupEquivariantCNN %>%
    luz::setup(
      loss = torch::nn_cross_entropy_loss(),
      optimizer = torch::optim_adam,
      metrics = list(
        luz::luz_metric_accuracy()
      )
    ) %>%
    luz::set_hparams(group = CyclicGroup(order = 4),
                     kernel_size = 5,
                     in_channels = 1,
                     out_channels = 10,
                     num_hidden = 4,
                     hidden_channels = 16) %>%
    luz::set_opt_hparams(lr = 1e-4, weight_decay = 1e-4) %>%
    luz::fit(train_dl, epochs = 1, valid_data = test_dl)

  # Making predictions ------------------------------------------------------

  preds <- predict(fitted, test_dl)

})
