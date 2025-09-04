#We will start by defining inputting the relevant parameters as given for distributions -------
mD=32 # Mean Demand for early discount fare, Poisson 
mR=18 # Mean Demand for regular fare, Poisson 
pD=156 # Discount Price in £
pR=195 #Regular Price in £
capacity=23 #Total capacity (All the rooms the hotel has)
#Let us now compute the hotel's expected revenue using the above information FCFS ---------------
#suppose the hotel admits on a first-come first-serve (FCFS) basis
ExpRevenue=rep(0,capacity+1)
for (i in 1:1) {
  #protect=i-1
  availforEarlyD=capacity #-protect; #protect is reserved for high fare, but as we operate on a FCFS basis we set our protection to 0
  ExpRevenue[i]=0;
  for(dD in 0:64){  #we use 64 as it is twice the mean for the discounted fare
    soldDiscount=min(availforEarlyD, dD)
    remainforReg=capacity-soldDiscount
    for (dR in 0:36){soldReg=min(remainforReg, dR) #we use 64 as it is twice the mean for the discounted fare
    RevenueThisIter=pD*soldDiscount+pR*soldReg #our revenue is a resutlt of the sum of the number of rooms sold at each price point multiplied by their respective price.
    ExpRevenue[i]=ExpRevenue[i]+
      RevenueThisIter*dpois(dD,mD)*dpois(dR, mR)}
  }
}
RevenueFCFS=ExpRevenue[1]
print(paste("Lower Bound for Expected Revenue (FCFS):", round(RevenueFCFS, 1)))

#Lets do the same with protection for late arrivals----
#we now wish to find out how many rooms we should keep for late arrivals, 
#those that do not book in advance at the discounted rate
ExpRevenue=rep(0,capacity+1)
for (i in 1:(capacity+1)){
  protect=i-1 
  availforEarlyD=capacity-protect; 
  ExpRevenue[i]=0;
  for(dD in 0:64){
    soldDiscount=min(availforEarlyD,dD) 
    remainforReg=capacity-soldDiscount
    for(dR in 0:36){
      soldReg=min(remainforReg,dR)
      RevenueThisIter=pD*soldDiscount+pR*soldReg
      ExpRevenue[i]=ExpRevenue[i]+
        RevenueThisIter*dpois(dD,mD)*dpois(dR,mR)
    }
  }
}

Protectindexbest = which(ExpRevenue == max(ExpRevenue)) 
ProtectBest=Protectindexbest-1
OptimalExpRevenue=max(ExpRevenue) #we want to maximise our returns by protecting rooms for those paying full fare, hence we use the max operator here
print(paste("The Optimal Protection Level for Regular-Fare Demand:", ProtectBest))
#lets now compute the expected revenue if we keep these rooms reserved for late arrival 
print(paste("The Expected Revenue When 14 Rooms are Reserved for Late Arrivals :", round(OptimalExpRevenue, 1)))
#lets compare this with our expected revenue from FCFS and compute the percentage change
pChange = ((OptimalExpRevenue - RevenueFCFS)/(RevenueFCFS))*100 #here we just calculate the percentage change numerically
print(paste('The Percentage Change From Moving From FCFS Basis to a Reserving for Regular Fare Basis is:', round(pChange, 1)))

#Sensitivity analysis-----
#here we want to explore the effect on expected revenue if we change the key parameters of our distribution
#We will start by changing the prices given, the regular price will be set at 200, consequently changing the discount price to 160
mD=32 # Mean Demand for early discount fare, Poisson 
mR=18 # Mean Demand for regular fare, Poisson 
pD=160 # Discount Price in £ this is changed as we changed the regular price 
pR=200 #Regular Price in £ this is what we have changed here 
#we will use the same code using the protection for regular fare paying customers and compare the expected revenues from that and this
ExpRevenue=rep(0,capacity+1)
for (i in 1:(capacity+1)){
  protect=i-1 
  availforEarlyD=capacity-protect; 
  ExpRevenue[i]=0;
  for(dD in 0:64){
    soldDiscount=min(availforEarlyD,dD) 
    remainforReg=capacity-soldDiscount
    for(dR in 0:36){
      soldReg=min(remainforReg,dR)
      RevenueThisIter=pD*soldDiscount+pR*soldReg
      ExpRevenue[i]=ExpRevenue[i]+
        RevenueThisIter*dpois(dD,mD)*dpois(dR,mR)
    }
  }
}

Protectindexbest = which(ExpRevenue == max(ExpRevenue))
ProtectBest=Protectindexbest-1
OptimalExpRevenue=max(ExpRevenue)
#lets now compute the expected revenue when we increase the price to £200  
print(paste("The Expected Revenue When The Price Increases to £200 :", round(OptimalExpRevenue, 1)))

#next we want to change the mean demand for each case, but we will revert back to the old revenue so we can compare it accurately
#here we will change the means one by one, first lets look at increasing the demand for discount, early reserve rooms 
mD=35 # Mean Demand for early discount fare, Poisson this is what we have changed here 
mR=18 # Mean Demand for regular fare, Poisson 
pD=156 # Discount Price in £
pR=195 #Regular Price in £
ExpRevenue=rep(0,capacity+1)
for (i in 1:(capacity+1)){
  protect=i-1 
  availforEarlyD=capacity-protect; 
  ExpRevenue[i]=0;
  for(dD in 0:64){
    soldDiscount=min(availforEarlyD,dD) 
    remainforReg=capacity-soldDiscount
    for(dR in 0:36){
      soldReg=min(remainforReg,dR)
      RevenueThisIter=pD*soldDiscount+pR*soldReg
      ExpRevenue[i]=ExpRevenue[i]+
        RevenueThisIter*dpois(dD,mD)*dpois(dR,mR)
    }
  }
}

Protectindexbest = which(ExpRevenue == max(ExpRevenue))
ProtectBest=Protectindexbest-1
OptimalExpRevenue=max(ExpRevenue)
print(paste("The Optimal Protection Level for Regular-Fare Demand:", ProtectBest))
#lets now compute the expected revenue when we increase the discount demand by 3   
print(paste("The Expected Revenue When we Increase the Mean Demand of Discounted Fare :", round(OptimalExpRevenue, 1)))


#finally, lets look at increasing the demand for regular rooms
mD=32 # Mean Demand for early discount fare, Poisson 
mR=20 # Mean Demand for regular fare, Poisson this is what we have changed here 
pD=156 # Discount Price in £
pR=195 #Regular Price in £
ExpRevenue=rep(0,capacity+1)
for (i in 1:(capacity+1)){
  protect=i-1 
  availforEarlyD=capacity-protect; 
  ExpRevenue[i]=0;
  for(dD in 0:64){
    soldDiscount=min(availforEarlyD,dD) 
    remainforReg=capacity-soldDiscount
    for(dR in 0:36){
      soldReg=min(remainforReg,dR)
      RevenueThisIter=pD*soldDiscount+pR*soldReg
      ExpRevenue[i]=ExpRevenue[i]+
        RevenueThisIter*dpois(dD,mD)*dpois(dR,mR)
    }
  }
}

Protectindexbest = which(ExpRevenue == max(ExpRevenue))
ProtectBest=Protectindexbest-1
OptimalExpRevenue=max(ExpRevenue)
print(paste("The Optimal Protection Level for Regular-Fare Demand:", ProtectBest))
#lets now compute the expected revenue when we increase the discount demand by 3   
print(paste("The Expected Revenue When we Increase the Mean Demand of Regular Fare :", round(OptimalExpRevenue, 1)))

