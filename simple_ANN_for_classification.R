#library(minpack.lm)
#library(mvtnorm)

#set.seed(42)
get_class_distribution <- function(fit, x1, x2) {
  test_grid <- expand.grid(x1 = x1, 
                           x2 = x2)
  matrix(round(predict(fit, test_grid)), nrow = length(x1))
}

# plot_decision_boundary <- function(df, x1, x2, z) {
#   #n <- 50
#   filled.contour(x = x1, y = x2, z = z,
#                  #nlevels = n,
#                  col = c("pink", "cyan"),#rainbow(n + 10, alpha = 0.5),#cm.colors(n+1, alpha = 0.5),
#                  plot.axes = {
#                    axis(1); axis(2);
#                    points(x = df$x1[df$y == 1],
#                           y = df$x2[df$y == 1],
#                           col = "blue")
#                    points(x = df$x1[df$y == 0],
#                           y = df$x2[df$y == 0], 
#                           col = "red")})
# }

#my_filled_contour <- function(x, y, z) {
  

  
  #return(invisible(NULL))
#}

plot_decision_boundary <- function(df, x, y, z, ...) {
  
  plot(x = NULL, y = NULL,
       xlim = range(x),
       ylim = range(y),
       xlab = "x1",
       ylab = "x2",
       frame = FALSE, axes = TRUE, 
       xaxs = "i", yaxs = "i",
       ...)
  
  .filled.contour(x = x, 
                  y = y, 
                  z = z,
                  levels = c(0,1,2),
                  col = c("pink", "cyan"))
  
  #contour(x,y,z,
  #        add = TRUE, 
  #        #lwd = 1,
  #        drawlabels = FALSE)
  
  box()
  
  #my_filled_contour(x1, x2, z)
  points(x = df$x1[df$y == 1],
         y = df$x2[df$y == 1],
         col = "blue")
  points(x = df$x1[df$y == 0],
         y = df$x2[df$y == 0], 
         col = "red")
  
  #return(invisible(NULL))
}

# create data
# max_data <- 500
# class1 <- rmvnorm(n = max_data, 
#                   mean = c(2.5, 2.5), 
#                   sigma = matrix(c(5, 1,
#                                    1, 5), 
#                                  ncol = 2))
# class2 <- rmvnorm(n = max_data, 
#                   mean = c(8.0, 5.0), 
#                   sigma = matrix(c(5, -2,
#                                    -2,  5), 
#                                  ncol = 2))
# 
# df <- data.frame(x1 = class1[,1], 
#                  x2 = class1[,2], 
#                  y = rep(1, nrow(class1)))
# df <- rbind(df, data.frame(x1 = class2[,1],
#                            x2 = class2[,2],
#                            y = rep(0, nrow(class2))))
# use some iris data
df <- subset(iris, 
             subset = Species != "setosa",
             select = c(Sepal.Length, Petal.Length, Species))
df$Sepal.Length <- as.vector(scale(df$Sepal.Length))
df$Petal.Length <- as.vector(scale(df$Petal.Length))
df$Species <- ifelse(df$Species == "versicolor", 1, 0)
colnames(df) <- c("x1", "x2", "y")
rownames(df) <- NULL

# test range
x1 <- seq(0.95*min(df$x1), 1.05*max(df$x1), 0.05)
x2 <- seq(0.95*min(df$x2), 1.05*max(df$x2), 0.05)
df_test_space <- expand.grid(x1 = x1,
                             x2 = x2)

# fit data with simple linear regression
fit1 <- lm(y ~ ., data = df)
pred_class <- ifelse(predict(fit1, 
                             newdata = df_test_space) < 0.5,
                     0, 1)
pred_class <- matrix(pred_class, nrow = length(x1))
plot_decision_boundary(df, x1, x2, pred_class, 
                       main = paste("linear fit, Acc = ",
                              mean(ifelse(predict(fit1) < 0.5, 0, 1) == df$y)))

# fit the data with logistic regression
fit2 <- glm(y ~ poly(x1, x2, degree = 1), 
            data = df, 
            family = binomial(link = "logit"))
pred_class <- round(predict(fit2, newdata = df_test_space, type = "response"))
pred_class <- matrix(pred_class, nrow = length(x1))
plot_decision_boundary(df, x1, x2, pred_class, 
                       main = paste("log reg fit, Acc = ",
                                    mean(round(predict(fit2, 
                                                       type = "response")) == df$y)))

# fit the data with a neural net by the library nnet
fit3 <- nnet::nnet(y ~ ., 
                   data = df, 
                   size = 2,
                   rang = 0.01,
                   #linout = TRUE,
                   maxit = 1000)
# plot fitted line
plot_decision_boundary(df, x1, x2, 
                       z = get_class_distribution(fit3, x1, x2),
                       main = paste("nnet, Acc =",
                                    mean(round(predict(fit3)) == df$y)))

#stop("BY USER")

# fit the data by my own ANN
act_func <- function(x) {
  #log(1.0 + exp(x))
  #1.0 / (1.0 + exp(-x))
  #tanh(x)
  exp(x)
  #ifelse(x < 0, 0, x)
  #ifelse(x < 0, 0, 1)
  #exp(-exp(-x))
  #sin(x)
}


nnet_function <- function(X, w) {
  # prepare weight matrixes
  W1 <- matrix(w[1:6], ncol = nrow(X)) # W$1
  W2 <- w[7:9] # W$2
  # compute neural network
  Z01 <- W1 %*% X
  Z1 <- rbind(1.0, act_func(Z01))
  W2 %*% Z1
}

y_test <- df$y
X <- t(model.matrix(y~., df))
loss_function <- function(w) {
  z <- nnet_function(X, w)
  p <- 1.0 / (1.0 + exp(-z))
  - mean(y_test*log(p) + (1 - y_test)*log(1 - p))
}

start_values_nnet <- rnorm(9)
names(start_values_nnet) <- paste0("w", seq_along(start_values_nnet))

fit4 <- optim(start_values_nnet, 
              loss_function,
              #method = "BFGS",
              #method = "CG",
              #method = "SANN", 
              method = "Nelder-Mead",
              control = list(trace = TRUE,
                             maxit = 1e5))


test <- t(model.matrix(~., data = df_test_space))
plot_decision_boundary(df, x1, x2, 
                       z = matrix(ifelse(nnet_function(test, fit4$par) < 0, 
                                         0, 1),
                                  ncol = length(x2)),
                       main = paste("my ANN, Acc =",
                              mean(ifelse(nnet_function(X, fit4$par) < 0, 
                                          0, 1) == df$y)))