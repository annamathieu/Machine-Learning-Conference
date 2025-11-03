# Préparation des répertoir de données 
# --- 1. Définition des paramètres ---
base_dir <- "C:\\Users\\rlmd2\\Documents\\5A\\Conf MLCS\\Pomme"
classes <- c("mures", "pourries")
split_ratio <- 0.8  # 80% pour l'entraînement, 20% pour le test
set.seed(42)        # Pour rendre le partage reproductible

# --- 2. Création des dossiers cibles s'ils n'existent pas ---

# Dossiers principaux
train_dir <- file.path(base_dir, "train")
test_dir <- file.path(base_dir, "test")

# Création des dossiers
dir.create(train_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(test_dir, recursive = TRUE, showWarnings = FALSE)

# --- 3. Boucle de séparation par classe ---
for (class in classes) {
  
  cat("Traitement de la classe:", class, "\n")
  
  # a. Définition des chemins complets
  source_path <- file.path(base_dir, class)
  train_class_path <- file.path(train_dir, class)
  test_class_path <- file.path(test_dir, class)
  
  # Création des sous-dossiers train/class et test/class
  dir.create(train_class_path, showWarnings = FALSE)
  dir.create(test_class_path, showWarnings = FALSE)
  
  # b. Liste de tous les fichiers (images) dans le dossier source
  all_files <- list.files(source_path, full.names = TRUE)
  num_files <- length(all_files)
  
  if (num_files == 0) {
    cat("  !!! Aucun fichier trouvé dans", source_path, "!!!\n")
    next
  }
  
  # c. Choix ALÉATOIRE des indices pour l'entraînement
  train_indices <- sample(seq_len(num_files), size = floor(split_ratio * num_files))
  
  # d. Détermination des fichiers train et test
  train_files <- all_files[train_indices]
  test_files <- all_files[-train_indices] # Le reste est pour le test
  
  # e. Déplacement des fichiers
  
  # Déplacement pour le TRAIN
  file.copy(from = train_files,
            to = file.path(train_class_path, basename(train_files)))
  
  # Déplacement pour le TEST
  file.copy(from = test_files,
            to = file.path(test_class_path, basename(test_files)))
  
  cat("  ->", length(train_files), "fichiers déplacés vers 'train/", class, "'\n", sep="")
  cat("  ->", length(test_files), "fichiers déplacés vers 'test/", class, "'\n", sep="")
}

cat("\nOpération de partage terminée. Vous pouvez maintenant exécuter votre code Keras.")

# Define the image size
image_size <- c(128, 128)
#Notre taille d'image est définie dans le kaggle donc on reprend la même 

# Load and preprocess image data
train_datagen <- image_data_generator(
  rescale = 1/255,                # Normalize pixel values to [0, 1]
  shear_range = 0.2,              # Randomly apply shear transformation
  zoom_range = 0.2,               # Randomly zoom into images
  horizontal_flip = TRUE          # Randomly flip images horizontally
)

test_datagen <- image_data_generator(
  rescale = 1/255                 # Only rescale for test data
)

# Load training and test data
train_directory <- "C:/Users/rlmd2/Documents/5A/Conf MLCS/Pomme/train"
test_directory <- "C:/Users/rlmd2/Documents/5A/Conf MLCS/Pomme/test"

train_generator <- flow_images_from_directory(
  train_directory,
  generator = train_datagen,
  target_size = image_size,
  batch_size = 32,
  class_mode = 'binary'  # For binary classification
)

test_generator <- flow_images_from_directory(
  test_directory,
  generator = test_datagen,
  target_size = image_size,
  batch_size = 32,
  class_mode = 'binary'
)

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = 'relu', 
                input_shape = c(128, 128, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  layer_flatten() %>%
  
  layer_dense(units = 512, activation = 'relu') %>%
  
  # Output layer with 1 unit for binary classification
  layer_dense(units = 1, activation = 'sigmoid')




# Print the model summary
summary(model)

model %>% compile(
  optimizer = optimizer_adam(),
  loss = 'binary_crossentropy',  # Use binary crossentropy for binary classification
  metrics = c('accuracy')
)

history <- model %>% fit(
  train_generator,
  steps_per_epoch = ceiling(23 / 32),  # 23 images / 32 batch size
  epochs = 20,
  validation_data = test_generator,
  validation_steps = ceiling(9 / 32)  # 9 images / 32 batch size
)

evaluation <- model %>% evaluate(
  test_generator,
  steps = as.integer(200 / 32)
)

cat('Test loss:', evaluation$loss, '\n')
cat('Test accuracy:', evaluation$accuracy, '\n')

# Plot loss curves
plot(history$metrics$loss, type = 'l', col = 'blue', xlab = 'Epoch', ylab = 'Loss',
     main = 'Training and Validation Loss')
lines(history$metrics$val_loss, type = 'l', col = 'red')
legend('topright', legend = c('Training Loss', 'Validation Loss'), 
       col = c('blue', 'red'), lty = 1)

# Plot accuracy curves
plot(history$metrics$accuracy, type = 'l', col = 'blue', xlab = 'Epoch', ylab = 'Accuracy', 
     main = 'Training and Validation Accuracy')
lines(history$metrics$val_accuracy, type = 'l', col = 'red')
legend('bottomright', legend = c('Training Accuracy', 'Validation Accuracy'), 
       col = c('blue', 'red'), lty = 1)

