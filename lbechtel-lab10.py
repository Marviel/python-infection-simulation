#!/usr/bin/env python3

from random import random as rand
import random
import time
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *
from threading import Thread


NROWS = 10
NCOLS = 10
TIMESTEP = 0 #in seconds
MAX_DAYS = 20
TAU = .4
NU = .01
THREADNUM = 10




class Simulation(object):
  def __init__(self, nrows, ncols, tau, nu, max_days):
    self.nrows = nrows #cols
    self.ncols = ncols #rows
    self.tau = tau # Infection rate (between 0 and 1)
    self.nu = nu # Vaccination rate (between 0 and 1)
    self.k = max_days # Max days

    self.grid = [[0 for cols in range(self.ncols)] for rows in range(self.nrows)]
    self.recovered_cnt = 0
    self.infected_cnt = 0
    self.vaccinated_cnt = 0
    self.infected_list = []
    self.recovered_list = []
    self.vaccinated_list = []
    self.days = 0

  #Steps the simulation. Returns the number of infected patients.
  def step(self):
    ninfected = 0
    #Iterate through all patients
    for r in range(self.nrows):
      for c in range(self.ncols):
        if self.grid[r][c] < 0: # patient is either vaccinated or recovered, nothing needs to be done. 
          continue
        elif self.grid[r][c] == 0:
          #If the patient is susceptible, call the infect() subroutine (see below) to determine if the patient
          #  becomes infected on this iteration of the simulation. 
          if self.roll_infect(r,c): 
            self.infect_patient(r,c)
          #If the patient has not become infected, the code should give him a chance to become vaccinated (using vaccination rate nu).
          elif self.roll_vaccinate(r,c): 
            self.vaccinate_patient(r,c)
        elif self.grid[r][c] > 0:
          ninfected += 1 #increase infected count
          # If the patient is infected (matrix value is a positive integer), increment the integer once
          self.grid[r][c] += 1
          # check if it has exceeded k (the predetermined maximum number of days the disease can last).
          if self.grid[r][c] > self.k:
            self.recover_patient(r,c) #If it has, set the value to -1 (recovered).

    self.infected_list.append(self.infected_cnt)
    self.recovered_list.append(self.recovered_cnt)
    self.vaccinated_list.append(self.vaccinated_cnt)

    self.days += 1
    return ninfected


  def roll_infect(self, r, c):
    t=False

    # there should be a better way to calculate these, but I don't know it yet.
    ds = [(1,0), (0, 1), (-1,0), (0,-1)]

    # Check all neighbors
    for dr, dc in ds:
      # print "r: %d, dr: %d"%(r, dr)
      # print "c: %d, dc: %d"%(c, dc)
      # print r + dr
      # print c + dc
      if self.check_valid_offset(r, c, dr, dc): # Is there a neighbor at this position in the grid?
        if (self.grid[r + dr][c + dc] > 0): # if this neighbor is infected
          if not t:
            t = (rand() < self.tau) # use a random number and the infection spread probability
                             # to determine if this patient will become infected

    # if after checking all neighbors the simulation determines that this
    # patient has become infected, set the value to "1" (meaning, "patient
    # in first day of infection").
    return t

  def roll_vaccinate(self, r, c):
    return rand() < self.nu

  #Checks if an offset is valid
  def check_valid_offset(self,r,c, dr, dc):
    newr = r + dr
    newc = c + dc

    if newr < 0 or newr >= self.nrows:
      return False
    if newc < 0 or newc >= self.ncols:
      return False

    return True

  # Infects a random patient in the grid.
  def infect_random(self):
    randr = random.randrange(self.nrows)
    randc = random.randrange(self.ncols)
    self.infect_patient(randr, randc)

  # Infects a patient, and increases the self.infected_cnt to reflect the increase.
  def infect_patient(self, r, c):
    self.grid[r][c] = 1
    self.infected_cnt += 1

  # Vaccinates a patient, and increases the self.vaccinated_cnt to reflect the increase.
  def vaccinate_patient(self, r, c):
    self.grid[r][c] = -2
    self.vaccinated_cnt += 1

  # Recovers a patient, and increases the self.recovered_cnt to reflect the increase.
  #  This should only happen after the disease runs its course.
  def recover_patient(self, r, c):
    self.grid[r][c] = -1
    self.recovered_cnt += 1

  # Prints the current contents of self.grid.
  def print_grid(self):
    for row in range(self.nrows):
      print(self.grid[row])

  def print_info(self):
    print("\tVaccinated Count: %d"%(self.vaccinated_cnt))
    print("\tInfected Count: %d"%(self.infected_cnt))
    print("\tRecovered Count: %d"%(self.recovered_cnt))

  # Calculates the correct color as (R,G,B) tuple for a cell given its value.
  def get_rgb_color(self, val):
    if val == -2: # Recovered ... a lotta green
      return [0,.6,0]
    elif val == -1: # Vaccinated .... a little green
      return [0,.3,0]
    elif val == 0: # Unaffiliated ... some blue
      return [0,0,.5]
    else: # Infected... a range of red values from 0 to 1
      norm = val/self.k
      return [norm,0,0]




# for i in range(0,1) :
#      current = ThreadTest()
#      current.start()
#      current.join()
#      #print ("\tThread ", i, " returnVal is: ", current.returnVal)


# The code that the thread executes goes here:
def run_sim_video(sim):
  sim.infect_random()
  fig = plt.figure(1)
  ax = fig.add_subplot(111)
  ax.set_title("Beds")

  im = ax.imshow(np.zeros((NROWS, NCOLS, 3)), interpolation='nearest') # Blank starting image
  fig.show()
  im.axes.figure.canvas.draw()

  x = 0
  while sim.step() > 0:
    print("x = %d ------------------------------------"%(x))
    sim.print_grid()

    H = np.array(sim.grid)[::]
    D = np.zeros((NROWS,NCOLS,3))
    for r in range(len(H)):
      for c in range(len(H[r])):
        D[r][c] = sim.get_rgb_color(H[r][c])

    ax.set_title( str( x ) )
    im.set_data( D )
    im.axes.figure.canvas.draw()

    x += 1
    time.sleep(TIMESTEP)

def run_sim(sim):
  sim.infect_random()

  x = 0
  while sim.step() > 0:
    #print("x = %d ------------------------------------"%(x))
    #sim.print_grid()

    H = np.array(sim.grid)[::]
    D = np.zeros((NROWS,NCOLS,3))
    for r in range(len(H)):
      for c in range(len(H[r])):
        D[r][c] = sim.get_rgb_color(H[r][c])

    x += 1
    time.sleep(TIMESTEP)



class ThreadTest(Thread):
  # Define any args you will be passing to the thread here:
  def __init__(self, sim):
    Thread.__init__(self)
    self.sim = sim

  # The code that the thread executes goes here:
  def run(self):
    run_sim(self.sim)



import sys



if THREADNUM > 1:
  vaccinated_cnt = 0
  infected_cnt = 0
  recovered_cnt = 0
  vaccinated_list = []
  infected_list = []
  recovered_list = []
  thread_list = []
  finished = False
  max_day_count = 0
  for i in range(0,THREADNUM):
    sim = Simulation(NROWS, NCOLS, TAU, NU, MAX_DAYS)
    current = ThreadTest(sim)
    thread_list.append(current)
    current.start()

  for t in thread_list:
    t.join()

  i = 0
  sys.stdout.flush()

  #Get max days of sims.
  for t in thread_list:
    if max_day_count < t.sim.days: max_day_count = t.sim.days

  #Summing all elements of arrays.
  vacc_sum_list = [0 for x in range(max_day_count)]
  for t in thread_list:
    print("====THREAD:  %d================================================================="%(i))
    print("  infected: %s"%(str(t.sim.infected_list)))
    print("  vaccinated: %s"%(str(t.sim.vaccinated_list)))
    print("  recovered: %s"%(str(t.sim.recovered_list)))

    vaccinated_cnt += t.sim.vaccinated_cnt
    infected_cnt += t.sim.infected_cnt
    recovered_cnt += t.sim.recovered_cnt

    # The elements of the array with the difference between the vacc_sum_list and vacc_list padded by the last element of vacc_list
    diff = len(vacc_sum_list) - len(t.sim.vaccinated_list)
    padded_vac = t.sim.vaccinated_list + [t.sim.vaccinated_list[-1] for x in range(diff)]
    print("erm----------")
    print(t.sim.vaccinated_list)
    print(padded_vac)
    print(len(padded_vac))
    print(len(vacc_sum_list))

    vacc_sum_list = [sum(x) for x in zip(padded_vac, vacc_sum_list)]
    print(vacc_sum_list)

    i += 1


  avg_vacc = 1.0*vaccinated_cnt/(1.0*THREADNUM)
  avg_inf = 1.0*infected_cnt/(1.0*THREADNUM)
  avg_rec = 1.0*recovered_cnt/(1.0*THREADNUM)
  print("========================================================")
  print("====================STATISTICS==========================")
  print("========================================================")
  print("  vacc_sum: %s"%(str(vacc_sum_list)))
  print("  vacc_cnt: %d"%(vaccinated_cnt))
  print("  inf_cnt: %d"%(infected_cnt))
  print("  rec_cnt: %d"%(recovered_cnt))
  print("  avg_vacc: %f"%(avg_vacc))
  print("  avg_inf: %f"%(avg_inf))
  print("  avg_rec: %f"%(avg_rec))
else:
  sim = Simulation(NROWS, NCOLS, TAU, NU, MAX_DAYS)
  run_sim_video(sim)
  # The total number of recovered patients at the end of the simulation instance. 
  sim.print_info()





# The goal of this assignment is to create a Monte Carlo simulator of the spread of an infectious disease in a hospital ward. 
#For simplicity, assume that the ward always contains 100 patients, arranged in a 10x10 (n=10) grid with equal distances between neighboring patients. 
#Assume that the infectious disease in this model always lasts exactly k days 
#  (meaning, every patient who gets it will recover after k days and will never be susceptible to that particular infection again). 
#In this model, a patient may only be infected by a neighboring patient. Throughout the simulation, a patient may be in one of four states: 
#     susceptible to infection, infected (in the ith day of infection; 1 <= i <= k), recovered, or vaccinated. Let these states be represented by 0,i,-1,-2, 
#     respectively. The Monte Carlo simulator will require the following parameters to determine state transitions for each grid location (patient):

# tau: Infection spread rate. The probability that a susceptible patient will become infected 
#   (Only relevant if at least one of the neighbors is infected). Possible values between 0 and 1.
# nu: Vaccination rate. Only susceptible patients can be vaccinated. Possible values between 0 and 1.
# Simulation Initialization

# At the start of the simulation, initialize the 10x10 matrix to all zeros. Randomly select one location and set it to 1 (assume there is only one infected patient at the start of the simulation). Use numpy.random.rand() whenever you need to generate one random number between 0 and 1. This function will be highly useful throughout the simulation.

# Simulation Model

# On every iteration and for every patient, first check the patient's status (matrix value). 
#If the patient is either vaccinated or recovered, nothing further needs to be done. 
#If the patient is susceptible, call the infect() subroutine (see below) to determine if the patient becomes infected on this iteration of the simulation. 
#If the patient has not become infected, the code should give him a chance to become vaccinated (using vaccination rate nu). 
#If the patient is infected (matrix value is a positive integer), increment the integer once and check if it has exceeded k 
#    If it has, set the value to -1 (recovered).


# Simulation Termination
# The simulation should terminate once all patients have recovered. 
# In terms of the values of the matrix, this means that the simulation should terminate when the matrix no longer contains any positive, 
#   non-zero values. Note that because of randomness, some of the simulation instances may be very brief.

# Infection Spread Algorithm

# In this simulation, three conditions must be met in order for a patient to become infected:

# Patient is currently susceptible (matrix cell value is zero)
# Patient has at least one infected neighbor. Having more than one infected neighbor should increase the probability of infection spread 
#    (for every patient, your python script should generate a random number and check against tau once for every one of that patient's infected neighbors).
# A random number generated by numpy.random.rand() is less than the infection spread rate tau.
# The following is partial code from an infection spread simulation subroutine. 
# It demonstrates how your simulation might check if patient (i,j) becomes infected: (numpy.random.rand() renamed rand() in this script)
 



                                                             
# The infect subroutine would then be called from the main simulation loop for every susceptible patient.
# Vaccination Algorithm

# In code, the vaccination algorithm will most likely look very similar to the infection spread algorithm. 
# The main difference is that in our simulation, vaccination is completely random. 
# The infection status of a patient's neighbors does not matter. 
# Every susceptible patient has an equal chance to be vaccinated on every iteration of the simulation. 
# Simply generate a random number and check against the vaccination probability nu.

# Results Summary







# Among the results produced by every simulation instance there should be:

# A vector containing the numbers of infected patients for every day of the simulation. For example, this may look like:
# infected: [2, 3, 4, 6, 9, 13, 14, 16, 14, 15, 13, 14, 12, 7, 5, 2, 1, 0]
# A vector containing the numbers of vaccinated patients for every day of the simulation. For example, this may look like:
# vaccinated: [10, 18, 24, 28, 31, 34, 34, 38, 41, 43, 45, 47, 49, 51, 51, 53, 54, 56]
# A vector containing the numbers of recovered patients for every day of the simulation. This might be:
# recovered: [0, 0, 0, 0, 2, 9, 10, 14, 20, 31, 39, 52, 64, 70, 77, 84, 84, 87]
# Using the three vectors above, we can determine that (for example), on day 5 of the simulation, 9 patients are infected, 31 have been vaccinated, and 2 have recovered. 
# Note that Susceptible Population = TotalPopulation - (recovered + infected + vaccinated). 
# In averaging vector elements (across all simulation threads), normalize by the number of threads and pad each vector using the last element of the array rather than zero. 
# Please write the three output vectors generated by each thread to an output file or
# to the console (stdout). The final set of vectors (averaged) should be written to file or stdout as well.

# Your simulation should use pylab plotting to summarize the results for k=20. Below are two example images:



