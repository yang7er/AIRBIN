# a function implementing the model
model <- function(hyperparameters){

  input <- layer_input(shape = c(hyperparameters$input_vars))

  for(i in 1:(hyperparameters$n_hidden_layers)){
    hidden_layers <- input %>%
      layer_dense(units = hyperparameters$init_filters/ (hyperparameters$filter_reduction_factor ^ (i - 1)), 
                  activation = 'relu',
                  kernel_regularizer = regularizer_l2(l = hyperparameters$lambda)) %>%
      layer_dropout(hyperparameters$init_dropout / (hyperparameters$dropout_reduction_factor ^ (i - 1)))
  }

  regression_head <- hidden_layers %>%
    layer_dense(units = 1)

  out_model <- keras_model(inputs = input, outputs = regression_head)

  out_model %>%
    compile(
      optimizer = optimizer_adam(learning_rate = hyperparameters$learning_rate),
      loss = 'mean_squared_error'
    )

  out_model
}