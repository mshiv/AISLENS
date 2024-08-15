library(sf)
library(rnaturalearth)
library(roughsf)

# Get Antarctica map data
antarctica <- ne_countries(scale = "medium", country = "Antarctica", returnclass = "sf")

# Define the polar stereographic projection for Antarctica
ant_proj <- "+proj=stere +lat_0=-90 +lat_ts=-71 +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

# Transform the data to the polar stereographic projection
antarctica_proj <- st_transform(antarctica, crs = ant_proj)

# Prepare the Antarctica polygon for roughsf
antarctica_proj <- st_cast(antarctica_proj, "POLYGON")
antarctica_proj$fill <- "#A0C4FF"  # Darker blue fill
antarctica_proj$stroke <- 2
antarctica_proj$fillweight <- 0.5

# Create the rough map
roughsf(list(antarctica_proj),
        caption = "Drawn with roughsf",
        title_font = "48px Pristina",
        font = "24px Pristina",
        caption_font = "24px Pristina",
        roughness = 1.5,
        bowing = 1,
        simplification = 1,
        width = 800,
        height = 800)