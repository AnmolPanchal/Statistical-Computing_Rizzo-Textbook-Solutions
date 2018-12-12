
 # import the library to test the normality of the distribution
library(stats)
size = 100000

u = runif(size)
v = runif(size)

x=rep(0,size)
y=rep(0,size)

for (i in 1:size){
  x[i] = sqrt(-2*log(u[i]))*cos(2*pi*v[i])
  y[i] = sqrt(-2*log(u[i]))*sin(2*pi*v[i])
}

#a test for normality
lillie.test(c(x,y))

#plot the estimation of the density
plot(density(c(x,y)))

num.samples <-  1000
U           <-  runif(num.samples)
X           <- -log(1-U)/2

# plot
hist(X, freq=F, xlab='X', main='Generating Exponential R.V.')
curve(dexp(x, rate=2) , 0, 3, lwd=2, xlab = "", ylab = "", add = T)
