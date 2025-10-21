# Test réduit sur les pommes 

require(keras3)

# # Create a simple sequential model
# model <- keras_model_sequential() %>%
#   layer_dense(units = 32, activation = 'relu', input_shape = c(784)) %>%
#   layer_dense(units = 10, activation = 'softmax')
# 
# # Print model summary
# summary(model)

# Test sur les pommes 

# Isolation des pommes
apples <- df_images %>% filter(y %in% c("RottenApple", "FreshApple"))


