library(sf)
library(roughsf)
library(ggplot2)
library(dplyr)

# Simulate sea level rise data with slower initial rise and increased uncertainty after 2100
set.seed(123)
years <- seq(2000, 2300, by = 10)
n_scenarios <- 50  # Increased number of scenarios

create_nonlinear_scenario <- function(initial_rate, acceleration_factor, uncertainty_factor) {
  initial_period <- which(years <= 2100)
  acceleration_period <- which(years > 2100)
  
  initial_rise <- cumsum(rnorm(length(initial_period), mean = initial_rate, sd = 0.05))
  
  accelerated_rise <- cumsum(rnorm(length(acceleration_period), 
                                   mean = initial_rate * acceleration_factor, 
                                   sd = 0.05 * uncertainty_factor) * 
                               (1 + (seq_along(acceleration_period) / length(acceleration_period))^2))
  
  c(initial_rise, initial_rise[length(initial_rise)] + accelerated_rise)
}

scenarios <- lapply(1:n_scenarios, function(i) {
  create_nonlinear_scenario(initial_rate = runif(1, 0.2, 0.5), 
                            acceleration_factor = runif(1, 1, 3),
                            uncertainty_factor = runif(1, 1, 4))
})

# Calculate mean scenario
mean_scenario <- Reduce(`+`, scenarios) / length(scenarios)

# Create sf objects for each scenario
create_sf_line <- function(years, values, scenario) {
  pts <- st_sfc(st_linestring(cbind(years, values)))
  st_sf(scenario = scenario, geometry = pts)
}

sf_scenarios <- lapply(1:n_scenarios, function(i) {
  create_sf_line(years, scenarios[[i]], paste("Scenario", i))
})

# Add mean scenario
sf_scenarios[[n_scenarios + 1]] <- create_sf_line(years, mean_scenario, "Mean")

# Combine sf objects
all_scenarios <- do.call(rbind, sf_scenarios)

# Set attributes for roughsf
all_scenarios$stroke <- ifelse(all_scenarios$scenario == "Mean", 3, 1)  # Thicker line for mean
all_scenarios$color <- ifelse(all_scenarios$scenario == "Mean", "#0066CC", "#B3D9FF")  # Dark blue for mean, light blue for others

# Create the rough plot
rough_plot <- roughsf(
  list(all_scenarios),
  roughness = 2,
  bowing = 1,
  simplification = 1,
  width = 800,
  height = 600,
  title_font = "36px Arial",
  font = "24px Arial",
  caption_font = "18px Arial"
)

# Display the plot
rough_plot