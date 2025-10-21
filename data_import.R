# Importation des données 

library(tidyverse)
library(magick)

path <- "C:/documents/Agrocampus/M2/Conference/Machine-Learning-Conference/Data/archive/Dataset"
extensions <- c("png", "jpg", "jpeg") # Formats d'images pris en charge

# Lister tous les fichiers correspondants dans les sous-dossiers
images <- list.files(
  path = path,
  pattern = paste0("\\.(", paste(extensions, collapse="|"), ")$"),
  recursive = TRUE,
  full.names = TRUE
)

# Extraire la "classe" (nom du dossier parent)
classes <- basename(dirname(images)) # fresh / rotten 

# Créer le data frame
df_images <- data.frame(
  x = images,
  y = classes,
  stringsAsFactors = FALSE
)

# Afficher un aperçu
head(df_images)


# Exportation du data set 
write.csv(x = df_images, file = "df_images.csv")






