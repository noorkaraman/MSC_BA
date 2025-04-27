### SMM641 Group Project

#Loading packages
library(dplyr)
library(nloptr)

#Loading the dataset
data <- read.csv("livecyclehireupdates.csv")

colnames(data) <- c("Station_id", "Latitude", "Longitude", "Name", "Bikes_available", 
                    "Station_capacity", "Empty_spaces", "Terminal_name", "Area", 
                    "Availability%", "HighDemand")

HDhire <- read.csv("2024HD_BikeHire.csv", sep = ",", header = TRUE) #Demand at 10 Busiest stations per quarter in London (2024)
LDhire <-read.csv("2024LD_BikeHire.csv", sep = ",", header = TRUE)

#Simulating data
N=nrow(data) 
head(data)

#Demand per station-----------------------------------------------
sum(data$HighDemand == 1)
summary(HDhire)
summary(LDhire)

#On average there are 90 days in a quarter so to find daily demand we divide by 90
#Converting data from quarterly demand to daily demand at high & low demand stations
avhigh_demand = mean(mean(HDhire$Q1) + mean(HDhire$Q2) + mean(HDhire$Q3) + mean(HDhire$Q4)) /90

avlow_demand = mean(mean(LDhire$Q1) + mean(LDhire$Q2) + mean(LDhire$Q3) + mean(LDhire$Q4)) / 90

#Poisson Distribution based on specificed mean & sd
hdN <- sum(data$HighDemand == 1)
ldN <- sum(data$HighDemand == 0)

set.seed(1234)
highdemandstations <- rpois(hdN, lambda = avhigh_demand)
lowdemandstations <- rpois(ldN, lambda = avlow_demand)

# Allocating demand based on if stations are considered high demand vs low demand
# If availability is below 25%, station is considered high demand
data$DailyDemand <- ifelse(data$HighDemand == 1, highdemandstations, lowdemandstations)

# Calculating Peak Demand as 60% of total daily demand
# Demand during Peak vs Non Peak time (assumption)
peak_ratio <- 0.65

# Demand across the day for 
data$PeakDemand <- round(data$DailyDemand * peak_ratio)
data$NonPeakDemand <- data$DailyDemand - data$PeakDemand

# Using information about current commuting costs // evidence to back this up
# Peak WTP
peak_wtp_mean <- 2.8
peak_wtp_sd <- 0.15
non_peak_wtp_mean <- 1.5
non_peak_wtp_sd <- 0.15

# Simulate WTP using normal distribution
data$PeakWTP <- rnorm(N, mean = peak_wtp_mean, sd = peak_wtp_sd)
data$NonPeakWTP <- rnorm(N, mean = non_peak_wtp_mean, sd = non_peak_wtp_sd) 

# Summarising Simulated Data
data2 <- data %>%
  select("Station_id", "HighDemand", "PeakDemand","NonPeakDemand","PeakWTP", "NonPeakWTP")

# PART 1A------------------------------------------------------
# Setting a Single Price across NonPeak / Peak Time 
# Chose this method as the each station had a different level of NP and P demand associated with it. 
# More effective than creating for loops

# Objective Function: Evaluate Revenue for Single Price
eval_f_single <- function(price) {
  totalDemand <- 0
  
  # Calculate demand for both Peak and NonPeak at the same price
  for (i in 1:nrow(data2)) {
    if (data2$NonPeakWTP[i] >= price) {
      totalDemand <- totalDemand + data2$NonPeakDemand[i]
    }
    if (data2$PeakWTP[i] >= price) {
      totalDemand <- totalDemand + data2$PeakDemand[i]
    }
  }
  
  # Calculate total revenue
  revenue <- price * totalDemand
  
  # Return negative revenue (for minimisation in optimisation)
  return(-revenue)
}

# Define constraints: Lower and upper bounds for price
lb <- 1      # Minimum price
ub <- 3.5    # Maximum price

# Initial guess for price
x0 <- 2

# Optimisation settings
opts <- list(
  "algorithm" = "NLOPT_LN_COBYLA",
  "xtol_rel" = 1.0e-8,
  "maxeval" = 1000
)

# Run optimisation for single price
result_single <- nloptr(
  x0 = x0, 
  eval_f = eval_f_single, 
  lb = lb, 
  ub = ub, 
  opts = opts
)

# Extract optimal price and revenue
optimal_price_single <- result_single$solution
optimal_revenue_single <- -result_single$objective

# Display results
print(paste0("Optimal Single Price: £", round(optimal_price_single, 2)))
print(paste0("Maximum Revenue: £", round(optimal_revenue_single, 2)))


## PART 1B------------------------------------------------------
# Setting a Two Prices Simulatenously across NonPeak / Peak Time 
totalNonPeakDemand <- 0
totalPeakDemand <- 0

# Objective Function: Evaluate Revenue
eval_f <- function(prices) {
  # Assign prices for Peak and NonPeak
  price_Peak <- prices[1]
  price_NonPeak <- prices[2]
  
  # Calculate demand for Peak and NonPeak
  for (i in 1:nrow(data2)) {
    if (data2$NonPeakWTP[i] > price_NonPeak) {
      # Add corresponding NonPeakDemand to total (works like cumulative sum)
      totalNonPeakDemand <- totalNonPeakDemand + data2$NonPeakDemand[i]
    }
  }
  
  for (i in 1:nrow(data2)) {
    if (data2$PeakWTP[i] > price_Peak) {
      # Add corresponding PeakDemand to total
      totalPeakDemand <- totalPeakDemand + data2$PeakDemand[i]
    }
  }
  # Calculate total revenue
  revenue <- (
    price_Peak * totalPeakDemand +
      price_NonPeak * totalNonPeakDemand
  )
  
  # Return negative revenue (e.g. Maximises revenue)
  return(-revenue)
}

eval_g_ineq <- function(prices) {
  price_Peak <- prices[1]
  price_NonPeak <- prices[2]
  # Non Peak Price < Peak Price
  constraint <- c(price_NonPeak - price_Peak)
  
  return(constraint)
  
}
# Set lower and upper bounds for prices
lb <- c(1, 1)      
ub <- c(3.5, 3.5)  

# Starting point for prices
x0 <- c(2, 1.5)

# Optimisation settings
opts <- list(
  "algorithm" = "NLOPT_LN_COBYLA",
  "xtol_rel" = 1.0e-8,
  "maxeval" = 1000
)

# Run optimisation
result <- nloptr(
  x0 = x0, 
  eval_f = eval_f, 
  lb = lb, 
  ub = ub, 
  eval_g_ineq = eval_g_ineq, 
  opts = opts
)

# Extract optimal prices and revenue
optimal_prices <- result$solution
optimal_revenue <- -result$objective
percentchange <- ((optimal_revenue - optimal_revenue_single) / optimal_revenue_single) * 100


# Display results
print(paste0("Optimal Peak Price: £", round(optimal_prices[1], 2)))
print(paste0("Optimal NonPeak Price: £", round(optimal_prices[2], 2)))
print(paste0("Maximum Revenue: £", round(optimal_revenue, 2)))
print(paste0("Percentage Change Compared to Baseline: ", round(percentchange, 2), "%"))
 








# PART 2---------------------------------------
# Implement a tax/discount rate across High and Low Demand Stations
# Improve the redistributed of bikes across different stations 
# Eligible for discount / required if the departure station is [High/Low]

# Filter high and low demand data
high_demand_data <- data %>% filter(HighDemand == 1)
low_demand_data <- data %>% filter(HighDemand == 0)

# Total Capacities
total_capacity_high <- sum(high_demand_data$Bikes_available)
total_capacity_low <- sum(low_demand_data$Bikes_available)
overall_cap <- total_capacity_high + total_capacity_low

base_price <- optimal_price_single  # identified in Part 1

# Objective Function to Maximise Utilisation
eval_capacity <- function(prices) {
  discount_rate <- prices[1]
  tax_rate <- prices[2]
  
  price_high <- base_price + (base_price * tax_rate)
  price_low <- base_price - (base_price * discount_rate)
  
  total_sold_high <- 0
  total_sold_low <- 0
  
  # High-Demand Stations
  for (i in 1:nrow(high_demand_data)) {  # Loop through each high-demand station
    # IF bikes available and users are willing to pay > high price
    while (high_demand_data$Bikes_available[i] > 0 && 
           (high_demand_data$PeakWTP[i] >= price_high || high_demand_data$NonPeakWTP[i] >= price_high)) {
      
      if (high_demand_data$PeakWTP[i] >= price_high) {
        # Calculate the number of bikes to sell based on the minimum of peak demand or bikes available
        bikes_sold <- min(high_demand_data$PeakDemand[i], high_demand_data$Bikes_available[i])
        total_sold_high <- total_sold_high + bikes_sold
        high_demand_data$Bikes_available[i] <- high_demand_data$Bikes_available[i] - bikes_sold
        
      } else if (high_demand_data$NonPeakWTP[i] >= price_high) {
        # Calculate the number of bikes to sell based on the minimum of non-peak demand or bikes available
        bikes_sold <- min(high_demand_data$NonPeakDemand[i], high_demand_data$Bikes_available[i])
        total_sold_high <- total_sold_high + bikes_sold
        high_demand_data$Bikes_available[i] <- high_demand_data$Bikes_available[i] - bikes_sold
      }
    }
  }
  
  # Low-Demand Stations
  for (i in 1:nrow(low_demand_data)) {  # Loop through each low-demand station
    # IF bikes available and users are willing to pay > low price
    while (low_demand_data$Bikes_available[i] > 0 && 
           (low_demand_data$PeakWTP[i] >= price_low || low_demand_data$NonPeakWTP[i] >= price_low)) {
      
      if (low_demand_data$PeakWTP[i] >= price_low) {
        # Calculate the number of bikes to sell based on the minimum of peak demand or bikes available
        bikes_sold <- min(low_demand_data$PeakDemand[i], low_demand_data$Bikes_available[i])
        total_sold_low <- total_sold_low + bikes_sold
        low_demand_data$Bikes_available[i] <- low_demand_data$Bikes_available[i] - bikes_sold
        
      } else if (low_demand_data$NonPeakWTP[i] >= price_low) {
        # Calculate the number of bikes to sell based on the minimum of non-peak demand or bikes available
        bikes_sold <- min(low_demand_data$NonPeakDemand[i], low_demand_data$Bikes_available[i])
        total_sold_low <- total_sold_low + bikes_sold
        low_demand_data$Bikes_available[i] <- low_demand_data$Bikes_available[i] - bikes_sold
      }
    }
  }
           
  
  overall_bikes_left <- (total_capacity_high + total_capacity_low) - (total_sold_high + total_sold_low)
  return(-overall_bikes_left)  # Return negative for minimisation
}

# Initial guess for discount and tax rates
x0 <- c(0.1, 0.1)  

# Define bounds for rates
lb <- c(0, 0)      
ub <- c(0.5, 0.5)  

# Optimisation settings
opts <- list(
  "algorithm" = "NLOPT_LN_COBYLA",
  "xtol_rel" = 1.0e-8,
  "maxeval" = 1000
)

# Run optimisation to find the optimal rates
result <- nloptr(
  x0 = x0,
  eval_f = eval_capacity,
  lb = lb,
  ub = ub,
  opts = opts
)

# Extract optimal rates and capacity
optimal_rates <- result$solution
optimal_capacity <- -result$objective
optimal_tax_rate <- optimal_rates[2]
optimal_discount_rate <- optimal_rates[1]

# Calculate prices with optimal rates
price_with_tax <- base_price + (base_price * optimal_tax_rate)
price_with_discount <- base_price - (base_price * optimal_discount_rate)

# Display results
print(paste0("Optimal Tax Rate for High-Demand Stations: ", round(optimal_tax_rate * 100, 2), 
             "% with a price (with Tax) of: £", round(price_with_tax, 2)))

print(paste0("Optimal Discount Rate for Low-Demand Stations: ", round(optimal_discount_rate * 100, 2), 
             "% with a price (with Discount) of: £", round(price_with_discount, 2)))

print(paste0("Increase bikes ustilisation by ", round(optimal_capacity, 2), 
             " bikes, equating to ", round((optimal_capacity / overall_cap) * 100, 2), "%"))


#Plots for interpretation in report ----
#define range of tax and discount rates for plot
tax_rates <- seq(0, 0.5, by = 0.01)
discount_rates <- seq(0, 0.5, by = 0.01)

#Function to calculate utilisation based on rates using Part 2 logic
calculate_utilisation <- function(discount_rate, tax_rate) {
  price_high <- base_price + (base_price * tax_rate)
  price_low <- base_price - (base_price * discount_rate)
  
  total_sold_high <- 0
  total_sold_low <- 0
  
  #High-Demand Stations
  for (i in 1:nrow(high_demand_data)) {
    while (high_demand_data$Bikes_available[i] > 0 && 
           (high_demand_data$PeakWTP[i] >= price_high || high_demand_data$NonPeakWTP[i] >= price_high)) {
      
      if (high_demand_data$PeakWTP[i] >= price_high) {
        bikes_sold <- min(high_demand_data$PeakDemand[i], high_demand_data$Bikes_available[i])
        total_sold_high <- total_sold_high + bikes_sold
        high_demand_data$Bikes_available[i] <- high_demand_data$Bikes_available[i] - bikes_sold
        
      } else if (high_demand_data$NonPeakWTP[i] >= price_high) {
        bikes_sold <- min(high_demand_data$NonPeakDemand[i], high_demand_data$Bikes_available[i])
        total_sold_high <- total_sold_high + bikes_sold
        high_demand_data$Bikes_available[i] <- high_demand_data$Bikes_available[i] - bikes_sold
      }
    }
  }
  
  #Low-Demand Stations
  for (i in 1:nrow(low_demand_data)) {
    while (low_demand_data$Bikes_available[i] > 0 && 
           (low_demand_data$PeakWTP[i] >= price_low || low_demand_data$NonPeakWTP[i] >= price_low)) {
      
      if (low_demand_data$PeakWTP[i] >= price_low) {
        bikes_sold <- min(low_demand_data$PeakDemand[i], low_demand_data$Bikes_available[i])
        total_sold_low <- total_sold_low + bikes_sold
        low_demand_data$Bikes_available[i] <- low_demand_data$Bikes_available[i] - bikes_sold
        
      } else if (low_demand_data$NonPeakWTP[i] >= price_low) {
        bikes_sold <- min(low_demand_data$NonPeakDemand[i], low_demand_data$Bikes_available[i])
        total_sold_low <- total_sold_low + bikes_sold
        low_demand_data$Bikes_available[i] <- low_demand_data$Bikes_available[i] - bikes_sold
      }
    }
  }
  
  #Total utilisation
  total_sold <- total_sold_high + total_sold_low
  total_capacity <- sum(data$Bikes_available)
  utilisation <- total_sold / total_capacity
  
  return(utilisation)
}

#Utilisation for each tax rate
utilisation_tax <- sapply(tax_rates, function(tax_rate) {
  calculate_utilisation(discount_rate = optimal_discount_rate, tax_rate = tax_rate)
})

#Utilisation for each discount rate
utilisation_discount <- sapply(discount_rates, function(discount_rate) {
  calculate_utilisation(discount_rate = discount_rate, tax_rate = optimal_tax_rate)
})

#Maximum utilisation
max_utilisation_tax <- max(utilisation_tax)
max_utilisation_discount <- max(utilisation_discount)

#Data frames for plotting
tax_data <- data.frame(TaxRate = tax_rates, utilisation = utilisation_tax)
discount_data <- data.frame(DiscountRate = discount_rates, utilisation = utilisation_discount)

#Plot utilisation against tax rate
tax_plot <- ggplot(tax_data, aes(x = TaxRate, y = utilisation)) +
  geom_line() +
  geom_point(data = tax_data[tax_data$utilisation == max_utilisation_tax, ], aes(x = TaxRate, y = utilisation), color = "red", size = 3) +
  labs(title = "Bike utilisation vs Tax Rate", x = "Tax Rate", y = "Bike utilisation") +
  theme_minimal()

#Plot utilisation against discount rate
discount_plot <- ggplot(discount_data, aes(x = DiscountRate, y = utilisation)) +
  geom_line() +
  geom_point(data = discount_data[discount_data$utilisation == max_utilisation_discount, ], aes(x = DiscountRate, y = utilisation), color = "red", size = 3) +
  labs(title = "Bike utilisation vs Discount Rate", x = "Discount Rate", y = "Bike utilisation") +
  theme_minimal()

#Print the plots
print(tax_plot)
print(discount_plot)


