# R-script discussed at the class on confidence intervals (2026-03-18)
# Class: SIMP59
# Teacher: Joost van de Weijer

# Objectives:
# - know what a standard error is
# - know what a confidence interval is
# - know what a sampling distribution is
# - know how to visually represent confidence intervals
# - know how to estimate confidence intervals in R
# - understand the principle of bootstrapping
# - understand the principle of Bayesian statistics

# Start with a sample
student_sample <- c(63,42,48,48,61,64,80,69,49,56,66,55,72,69,59,55,37,58,60,55)
mean(student_sample)

# Create a population
student_population <- sample(20:100,size=500000,replace=TRUE,dnorm(seq(-4,4,length.out=81)))
mean(student_population)
sd(student_population)

# Take two samples from the population
sample1 <- sample(student_population,size=20)
mean(sample1)
sample2 <- sample(student_population,size=20)
mean(sample2)

# Take 1000 samples from the population to make a distribution of mean values
sample_means <- replicate(1000,mean(sample(student_population,size=20)))
hist( sample_means)
summary(sample_means)
SE <- sd(sample_means)
mean(sample_means) + 1.96 * SE
mean(sample_means) - 1.96 * SE

# Back to the original sample
SE <- sd(student_sample) / sqrt( 20 )
mean(student_sample) + 1.96 * SE
mean(student_sample) - 1.96 * SE

# t instead of z
qt( p = 0.975, df = 19)

# bootstrapping illustration
bootstrap_sample <- sample( student_sample, size = 20, replace = TRUE)
bootstrapped_sampling_distribution <- replicate(1000, mean(sample(student_sample, size=20,replace=TRUE)))
summary(bootstrapped_sampling_distribution)

# Bayesian illustration
# Starts with a probability distribution of mean values, e.g., all values from 20 to 100 equally likely.
# Then evaluates for each of the mean values in that distribution what the probability of the 
# data is. 

# Plot of the sample mean with error bars
my_data <- data.frame( sample_mean = mean(student_sample), upper = mean(student_sample) + 1.96 * SE,
                       lower = mean(student_sample) - 1.96 * SE, respondents = "students")
ggplot( my_data, aes(x = respondents, y = sample_mean)) + geom_point() +
   geom_errorbar(aes(ymin = lower, ymax = upper, width = 0)) + 
   scale_y_continuous(limits=c(20,100))

# A second sample to illustrate the confidence interval of the difference
working_sample <- c(43,44,75,56,66,56,53,62,71,61,57,55,45,67,44,51,44,53,61,44)
my_data <- data.frame(response=c(student_sample,working_sample),class=c(rep("student",20),rep("working",20)))

confint(lm(response ~ class,my_data))
ggplot(data.frame( difference = -2.9, lower = -9.356216, upper = 3.556216, group = "working"),
       aes(x="group",y=difference))+geom_point()+geom_errorbar(aes(ymin=lower,ymax=upper))

