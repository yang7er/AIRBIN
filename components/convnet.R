model <- function(hyperparameters){
  lagged_input_shape <- c(NULL, hyperparameters$n_lags, 1)
  external_input_shape <- c(NULL, hyperparameters$n_external_features)

  # 1D-Convolution model for time series
  input_lagged <- layer_input(shape=lagged_input_shape, name="timeseries_input")
  conv_out <- input_lagged %>% 
      layer_conv_1d(filters = hyperparameters$conv_filters, 
                    kernel_size = hyperparameters$conv_kernel_size, 
                    kernel_regularizer = regularizer_l2(l = hyperparameters$regularize_lambda),
                    activation = 'relu') %>%
      layer_max_pooling_1d(pool_size = hyperparameters$pool_size) %>% 
      layer_flatten() %>%
      layer_dense(units = hyperparameters$conv_dense_units, activation='relu')

  # External features (future events) input
  input_external <- layer_input(shape=external_input_shape, name="external_input")
  external_out <- input_external

  for(i in 1:(hyperparameters$n_hidden_layers)){
      external_out %<>%
        layer_dense(units = hyperparameters$init_filters/ (hyperparameters$filter_reduction_factor ^ (i - 1)), 
                    activation = 'relu') %>%
        layer_dropout(hyperparameters$init_dropout / (hyperparameters$dropout_reduction_factor ^ (i - 1)))
  }
  external_out %<>% 
      layer_dense(units = hyperparameters$conv_dense_units)

  # Dense layer for regression
  merged <- layer_add(c(conv_out, external_out))
  output <- merged %>%
      layer_dense(units=32, activation='relu') %>% 
      layer_dropout(rate=0.1) %>% 
      layer_dense(units=16, activation='relu') %>%
      layer_normalization() %>%
      layer_dense(units=1)

  # Compile model
  model <- keras_model(inputs=c(input_lagged, input_external), outputs=output)

  model %>% compile(
    loss='mean_squared_error',
    optimizer=optimizer_adam(learning_rate=hyperparameters$learning_rate)
  )
}