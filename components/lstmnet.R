model <- function(hyperparameters){

  lagged_input_shape <- c(NULL, hyperparameters$n_lags, 1)
  external_input_shape <- c(NULL, hyperparameters$n_external_features)

  # LSTM model for time series
  input_lagged <- layer_input(shape=lagged_input_shape, name="timeseries_input")
  lstm_out <- input_lagged %>% 
    layer_lstm(units = hyperparameters$lstm_filters, 
               return_sequences = F,
               kernel_regularizer = regularizer_l2(l = hyperparameters$regularize_lambda)
               )

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
    layer_dense(units = hyperparameters$lstm_filters)

  # Dense layer for regression
  merged <- layer_add(c(lstm_out, external_out))
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