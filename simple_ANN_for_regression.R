library(minpack.lm)

# set data function
ref_func <- function(x, a, b, c) {
  a*x^2 + b*x + c
}

# craete data, and plot it
#set.seed(42)
x <- seq(0, 10, 0.1)
y_test <- ref_func(x, -1, 10, 1) + rnorm(length(x))
df <- data.frame(y = y_test, x = x)
plot(x, y_test)
legend("topleft",
       inset = 0.01,
       legend = c("linear", "nnet", "ppr", "my ANN 1", "my ANN 2"),
       col = c("black", "red", "green", "blue", "magenta"),
       lty = 1)

# fit the data with the right model
fit1 <- lm(y ~ I(x^2) + x,
           data = df)
lines(x, fit1$fitted.values, 
      col = "black")

# fit the data with a neural net by the library nnet
fit2 <- nnet::nnet(y ~ x, 
                   data = df, 
                   size = 2,
                   rang = 0.01,
                   linout = TRUE,
                   maxit = 1000)
# plotfitted line
lines(x, fit2$fitted.values, col = "red")

# fit model by projection pursuit regression
fit3 <- ppr(y ~ x,
            data = df,
            nterms = 2)
lines(x, fit3$fitted.values, col = "green")

# fit the data by my own ANN
act_func <- function(x) {
  cos(x)
  #sin(x)
  #log(1.0 + exp(x))
  #x * tanh(log(1.0 + exp(x)))
  #1.0 / (1.0 + exp(-x))
  #tanh(x)
  #exp(x)
  #ifelse(x < 0, 0, 1)
  #ifelse(x < 0, 0, x)
  #ifelse(x < 0, 0.01 * x, x)
  #exp(-exp(-x))
}


nnet_function <- function(X, w) {
  # prepare weight matrixes
  W1 <- matrix(w[1:6], ncol = 3) # W$1
  W2 <- matrix(w[7:10], ncol = 1) # W$2
  # compute neural network
  Z01 <- X %*% W1
  Z1 <- cbind(1.0, act_func(Z01))
  Z1 %*% W2
}

X <- model.matrix(y ~ x, data = df) # get design matrix
loss_function <- function(w) {
  sum((y_test - nnet_function(X, w))^2)
}

# fit my ANN with one of the following algorithms
# simulated annealing is fine, 
# as it can handle very often strange starting values 
start_values_nnet <- rnorm(10)
names(start_values_nnet) <- paste0("w", seq_along(start_values_nnet))
fit4 <- optim(start_values_nnet, 
              loss_function,
              #method = "BFGS",
              #method = "CG",
              #method = "SANN", 
              method = "Nelder-Mead",
              control = list(trace = TRUE,
                             maxit = 1000))
# plot fitted line
lines(x, nnet_function(X, fit4$par), col = "blue")

# use nls.lm for fitting
residual_func <- function(w) {
  y_test - nnet_function(X,w)
}

start_values_nlslm <- fit4$par
fit5 <- nls.lm(par = start_values_nlslm,
               fn = residual_func,
               control = list(maxiter = 1024,
                              maxfev = 10000,
                              nprint = 100))
lines(x, nnet_function(X, fit5$par), col = "magenta")

print(fit2$wts)
print(fit4$par)
print(fit5$par)