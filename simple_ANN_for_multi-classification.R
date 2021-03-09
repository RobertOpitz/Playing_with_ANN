#library(minpack.lm)
#library(mvtnorm)
library(nnet)
library(compiler)

#set.seed(42)
#get_class_distribution <- function(fit, x1, x2) {
#  test_grid <- expand.grid(x1 = x1, 
#                           x2 = x2)
#  matrix(round(predict(fit, test_grid)), nrow = length(x1))
#}

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
df_x <- subset(iris, select = -Species)
for (i in seq(ncol(df_x)))
    df_x[,i] <- as.numeric(scale(df_x[,i]))
   
df_y <- NULL
for (species in levels(iris$Species))
  df_y <- cbind(df_y,
                ifelse(iris$Species == species, 1, 0))
df_y <- as.data.frame(df_y)
colnames(df_y) <- levels(iris$Species)

# fit the data with logistic regression
fit1 <- multinom(Species ~ ., 
                 data = cbind(df_x, Species = iris$Species),
                 maxit = 1000)

#pred_class <- round(predict(fit2, newdata = df_test_space, type = "response"))
#pred_class <- matrix(pred_class, nrow = length(x1))
#plot_decision_boundary(df, x1, x2, pred_class, 
#                       main = paste("log reg fit, Acc = ",
#                                    mean(round(predict(fit2, 
#                                                       type = "response")) == df$y)))

# fit the data with a neural net by the library nnet
fit2 <- nnet(x = df_x, 
             y = df_y, 
             size = 3,
             rang = 0.01,
             softmax = TRUE,
             maxit = 5000)

# plot fitted line
#plot_decision_boundary(df, x1, x2, 
#                       z = get_class_distribution(fit3, x1, x2),
#                       main = paste("nnet, Acc =",
#                                    mean(round(predict(fit3)) == df$y)))

#stop("BY USER")

# fit the data by my own ANN
act_func <- function(x) {
  #log(1.0 + exp(x))
  #1.0 / (1.0 + exp(-x))
  #tanh(x)
  #exp(x)
  #ifelse(x < 0, 0, 1)
  #ifelse(x < 0, 0, x)
  #ifelse(x < 0, 0, 0.01 * x)
  exp(-exp(-x))
  #sin(x)
  #cos(x)
}

nnet_function <- function(X, w) {
  # prepare weight matrixes
  W1 <- matrix(w[1:15], ncol = 3) # W$1
  W2 <- matrix(w[16:27], ncol = 3) # W$2
  # compute neural network
  #Z01 <- X %*% W1
  #Z1 <- cbind(1.0, act_func(Z01))
  Z1 <- cbind(1.0, act_func(X %*% W1))
  Z1 %*% W2
}

loss_function <- function(w) {
  y <- nnet_function(X, w)
  #softmax
  p <- exp(y)
  p <- p / rowSums(p)
  # cross-entropy loss function
  - sum(df_y * log(p))
}

X <- model.matrix(~., df_x)
df_y <- df_y
start_values_nnet <- rnorm(27)
names(start_values_nnet) <- paste0("w", seq_along(start_values_nnet))

start <- Sys.time()
fit3 <- optim(start_values_nnet, 
              loss_function,
              method = "BFGS",
              #method = "CG",
              #method = "SANN", 
              #method = "Nelder-Mead",
              control = list(trace = TRUE,
                             maxit = 20000))
print(Sys.time() - start)

pred <- nnet_function(X, fit3$par)
lev <- levels(iris$Species)
pred_class <- apply(pred, 1, function(x) lev[which.max(x)])
cat("MULTINOM: Train Acc. = ", 
    round(mean(predict(fit1) == iris$Species), 3), "\n")
cat("NNET: Train Acc. = ", 
    round(mean(predict(fit2, type = "class") == iris$Species),3), "\n")
cat("MY NNET: Train Acc. = ", 
    round(mean(pred_class == iris$Species), 3), "\n")

#test <- t(model.matrix(~., data = df_test_space))
#plot_decision_boundary(df, x1, x2, 
#                       z = matrix(ifelse(nnet_function(test, fit4$par) < 0, 
#                                         0, 1),
#                                  ncol = length(x2)),
#                       main = paste("my ANN, Acc =",
#                              mean(ifelse(nnet_function(X, fit4$par) < 0, 
#                                          0, 1) == df$y)))


get_acc <- function(x, y, w, lev) {
  pred <- nnet_function(x, w)
  pred_class <- apply(pred, 1, function(this) lev[which.max(this)])
  mean(pred_class == y)
}

get_loss_func <- function(x, y) {
  x; y
  function(w) {
    z <- nnet_function(x, w)
    #softmax
    p <- exp(z)
    p <- p / rowSums(p)
    # cross-entropy loss function
    - sum(y * log(p))
  }
}

df_x <- subset(iris, select = -Species)
start_values_nnet <- fit3$par
lev <- levels(iris$Species)
train_acc <- rep(NA, nrow(iris))
test_acc <- rep(NA, nrow(iris))
# LOOCV
for (i in seq(nrow(iris))) {
  # split train and test data
  train_x <- df_x[-i,]
  train_y <- df_y[-i,]
  test_x <- df_x[i,]
  test_y <- df_y[i,]
  # pre-processing
  preproc <- caret::preProcess(train_x, method = c("center", "scale"))
  train_x <- predict(preproc, train_x)
  test_x <- predict(preproc, test_x)
  train_x <- model.matrix(~., train_x)
  test_x <- model.matrix(~., test_x)
  loss_function <- get_loss_func(train_x, train_y)
  # training
  trained_model <- optim(start_values_nnet, 
                         loss_function,
                         method = "BFGS",
                         #method = "CG",
                         #method = "SANN", 
                         #method = "Nelder-Mead",
                         control = list(trace = TRUE,
                                        maxit = 500))
  # predictions
  train_acc[i] <- get_acc(train_x, iris$Species[-i], trained_model$par, lev)
  test_acc[i] <- get_acc(test_x, iris$Species[i], trained_model$par, lev)
}

cat("Mean Train Acc = ", round(mean(train_acc), 3), "\n")
cat("Mean Test Acc  = ", round(mean(test_acc), 3), 
    " (", round(sd(test_acc) / sqrt(length(test_acc)),3), ")\n")