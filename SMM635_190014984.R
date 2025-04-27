#Setup ----
#Import relevant libraries 
library(tidyverse)
library(lubridate)
library(stringr)

#Load and prepare data
reviews <- read_csv("platform_economy_reviews.csv")
mobility <- read_csv("platform_economy_mobility.csv")

#Ensure 'time' is in Date format
reviews <- reviews %>%
  mutate(time = ym(time))

mobility <- mobility %>%
  mutate(time = ym(time))

#Plot 1 ----

#Aggregate data for Plot 1
reviews_summary <- reviews %>%
  group_by(time, location) %>%
  summarise(
    avg_bad_service = mean(bad_service, na.rm = TRUE),
    avg_bad_food = mean(bad_food, na.rm = TRUE)
  )

#Plot 1 
ggplot(reviews_summary, aes(x = time)) +
  #Key variables for line graph
  geom_line(aes(y = avg_bad_service, color = "Service"), size = 1) +
  geom_line(aes(y = avg_bad_food, color = "Food"), size = 1) +
  #Add trend lines for readability and interpretation
  geom_smooth(aes(y = avg_bad_service, color = "Service"), method = "loess", se = FALSE, linetype = "dashed", size = 0.8) +
  geom_smooth(aes(y = avg_bad_food, color = "Food"), method = "loess", se = FALSE, linetype = "dashed", size = 0.8) +
  #Event markers for readability and interpretation
  geom_vline(xintercept = as.Date("2016-05-01"), color = "#D55E00", linetype = "longdash", size = 0.8) + 
  geom_vline(xintercept = as.Date("2017-05-01"), color = "#009E73", linetype = "longdash", size = 0.8) + 
  annotate("text", x = as.Date("2016-04-01"), y = 0.132, label = "Uber & Lyft Exit", color = "#D55E00", size = 3, fontface = "italic", angle=0) +
  annotate("text", x = as.Date("2017-04-01"), y = 0.132, label = "Uber & Lyft Return", color = "#009E73", size = 3, fontface = "italic", angle=0) +
  #Add labels
  labs(
    title = "Impact of Uber/Lyft Exit on Service and Food Quality",
    subtitle = "Negative review percentages over time, faceted by location",
    x = "Time",
    y = "Average Negative Reviews",
    color = "Review Type"
  ) +
  #Faceting over location
  facet_wrap(~location, ncol = 1, labeller = labeller(location = function(x) str_to_title(x))) +
  #Theme and color adjustments for aesthetics 
  scale_color_manual(values = c("Service" = "#0072B2", "Food" = "#E69F00")) + #Custom colors
  scale_x_date(labels = date_format("%b %Y"), breaks = "6 months") + #Better date formatting
  ylim(0.12, 0.25) + #Set y-axis limits for aesthetics 
  theme_minimal(base_size = 14) + #Increased base font size for readability 
  theme(
    plot.title = element_text(face = "bold", size = 14, hjust = 0.5), #Centered and bold title
    plot.subtitle = element_text(size = 12, hjust = 0.5), 
    strip.text = element_text(size = 14, face = "bold", color = "darkblue"), #Facet title style
    legend.position = "right", #Legend on RHS
    legend.title = element_text(size=12, face = "bold"), 
    legend.text = element_text(size = 10), 
    panel.grid.major = element_line(size = 0.2, color = "gray85"), #Softer gridlines to save ink
    panel.grid.minor = element_blank(), #Remove minor gridlines to save ink
    axis.text = element_text(size = 10), #Adjust axis text size for readability 
    axis.title = element_text(size = 12), #Adjust axis title size for readability 
    axis.text.x = element_text(angle = 45, hjust = 1)
  )

#Plot 2 ----

#Aggregate data for Plot 2
mobility_summary <- mobility %>%
  group_by(time, dma) %>%
  summarise(
    avg_wage = mean(avg_hourly_wage, na.rm = TRUE),
    quit_rate = mean(quit, na.rm = TRUE),
    avg_tenure = mean(tenure, na.rm = TRUE)
  )

#Plot 2
ggplot(mobility_summary, aes(x = time)) +
  #Key variables for line graph 
  geom_line(aes(y = avg_wage, color = "Average Hourly Wage ($)"), size = 1) +
  geom_line(aes(y = quit_rate * 100, color = "Quit Rate (%)"), size = 1) +
  geom_line(aes(y = avg_tenure, color = "Average Tenure (Months)"), size = 1) +
  #Event markers for readability and interpretation
  geom_vline(xintercept = as.Date("2016-05-01"), color = "#D55E00", linetype = "longdash", size = 0.8) +
  geom_vline(xintercept = as.Date("2017-05-01"), color = "#009E73", linetype = "longdash", size = 0.8) +
  annotate("text", x = as.Date("2016-02-01"), y = max(mobility_summary$avg_wage, na.rm = TRUE) + 5, 
           label = "Uber & Lyft Exit", color = "#D55E00", size = 4, fontface = "italic") +
  annotate("text", x = as.Date("2017-02-01"), y = max(mobility_summary$avg_wage, na.rm = TRUE) + 5, 
           label = "Uber & Lyft Return", color = "#009E73", size = 4, fontface = "italic") +
  #Add labels
  labs(
    title = "Worker Trends in Austin and Dallas",
    subtitle = "Hourly wages, quit rates, and tenure over time, faceted by location",
    x = "Time",
    y = "Metrics (Wage($), Quit Rate(%), Tenure(Mts))",
    color = "Metric"
  ) +
  #Faceting over location for readability and interpretation
  facet_wrap(~dma, ncol = 1, labeller = labeller(dma = function(x) str_to_title(x))) +
  #Theme and color adjustments for aesthetics 
  scale_color_manual(values = c(
    "Average Hourly Wage ($)" = "#0072B2", 
    "Quit Rate (%)" = "#E69F00", 
    "Average Tenure (Months)" = "#56B4E9"
  )) + 
  scale_x_date(labels = date_format("%b %Y"), breaks = "6 months") + #Better date formatting
  theme_minimal(base_size = 14) + #Increased base font size for readability 
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5), # Centered and bold title for aesthetics 
    plot.subtitle = element_text(size = 12, hjust = 0.5), 
    strip.text = element_text(size = 14, face = "bold", color = "darkblue"), #Facet title for aesthetics 
    legend.position = "top", #Legend at the top due to large title and metrics on y axis 
    legend.title = element_text(face = "bold"), 
    legend.text = element_text(size = 12), 
    panel.grid.major = element_line(size = 0.5, color = "gray85"), #Softer gridlines to save ink
    panel.grid.minor = element_blank(), #Remove minor gridlines to save ink
    axis.text = element_text(size = 12), #Adjust axis text size for readability 
    axis.text.x = element_text(angle = 45, hjust = 1), #Angled x-axis text for readability 
    axis.title = element_text(size = 14) # Adjust axis title size for readability 
  )
