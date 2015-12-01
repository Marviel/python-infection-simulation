#!/usr/bin/env python3
#Title: lbechtel-hw9.py
#Author:        Luke Bechtel
#Class:         CS370
#Professor:     Dr. Berry
#Description:   
# A monte carlo simulation of an infection throughout a hospital.
# WILL PRODUCE VIDEO IF THREADNUM ARGUMENT IS SET TO 1
# Details for running are found by running: ./lbechtel-lab10.py

from random import random as rand
from random import randrange
#import random
import time
import numpy as np
import matplotlib as ml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *
from threading import Thread
import sys

if len(sys.argv) != 7:
  print("USAGE: ./lbechtel-lab10 [NROWS: int, number of rows.] [NCOLS: int, number of cols.] [MAX_DAYS: int, days that the virus will die out in] [TAU: infection rate] [NU: vaccination rate] [THREADNUM: Number of threads. 1 for video.]")
  sys.exit(0)

TIMESTEP = .1 #in seconds
NROWS = int(sys.argv[1])
NCOLS = int(sys.argv[2])
MAX_DAYS = int(sys.argv[3])
TAU = float(sys.argv[4])
NU = float(sys.argv[5])
THREADNUM = int(sys.argv[6])

#Contains data related to the monte carlo simulation.
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

  #Check if a cell should be infected.
  def roll_infect(self, r, c):
    t=False

    # there should be a better way to calculate these, but I don't know it yet.
    ds = [(1,0), (0, 1), (-1,0), (0,-1)]

    # Check all neighbors
    for dr, dc in ds:
      if self.check_valid_offset(r, c, dr, dc): # Is there a neighbor at this position in the grid?
        if (self.grid[r + dr][c + dc] > 0): # if this neighbor is infected
          if not t:
            t = (rand() < self.tau) # use a random number and the infection spread probability
                             # to determine if this patient will become infected

    # if after checking all neighbors the simulation determines that this
    # patient has become infected, set the value to "1" (meaning, "patient
    # in first day of infection").
    return t

  #Check if a cell should be vaccinated.
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
    randr = randrange(self.nrows)
    randc = randrange(self.ncols)
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
    self.infected_cnt -= 1
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

    #Colors for video.
    H = np.array(sim.grid)[::]
    D = np.zeros((NROWS,NCOLS,3))
    for r in range(len(H)):
      for c in range(len(H[r])):
        D[r][c] = sim.get_rgb_color(H[r][c])

    #Place data for video.
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



class ThreadTest(Thread):
  # Define any args you will be passing to the thread here:
  def __init__(self, sim):
    Thread.__init__(self)
    self.sim = sim

  # The code that the thread executes goes here:
  def run(self):
    run_sim(self.sim)


import sys


# If we're looking at multiple threads, we don't show the video.
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

  #list of total number of each type seen on a given day.
  vacc_sum_list = [0 for x in range(max_day_count)]
  rec_sum_list = [0 for x in range(max_day_count)]
  inf_sum_list = [0 for x in range(max_day_count)]
  for t in thread_list:
    print("====THREAD:  %d================================================================="%(i))
    print("  infected: %s"%(str(t.sim.infected_list)))
    print("  vaccinated: %s"%(str(t.sim.vaccinated_list)))
    print("  recovered: %s"%(str(t.sim.recovered_list)))

    vaccinated_cnt += t.sim.vaccinated_cnt
    infected_cnt += t.sim.infected_cnt
    recovered_cnt += t.sim.recovered_cnt

    # Add each day's totals to the summed totals of all threads.
    diff = len(vacc_sum_list) - len(t.sim.vaccinated_list)
    padded = t.sim.vaccinated_list + [t.sim.vaccinated_list[-1] for x in range(diff)]
    vacc_sum_list = [sum(x) for x in zip(padded, vacc_sum_list)]
    
    # Add each day's totals to the summed totals of all threads.
    diff = len(inf_sum_list) - len(t.sim.infected_list)
    padded = t.sim.infected_list + [t.sim.infected_list[-1] for x in range(diff)]
    inf_sum_list = [sum(x) for x in zip(padded, inf_sum_list)]

    # Add each day's totals to the summed totals of all threads.
    diff = len(rec_sum_list) - len(t.sim.recovered_list)
    padded = t.sim.recovered_list + [t.sim.recovered_list[-1] for x in range(diff)]
    rec_sum_list = [sum(x) for x in zip(padded, rec_sum_list)]

    i += 1

  #Calculate average of each important variable per day throughout all simulations.
  avg_vacc_list = np.divide(np.array(vacc_sum_list), THREADNUM)
  avg_rec_list = np.divide(np.array(rec_sum_list), THREADNUM)
  avg_inf_list = np.divide(np.array(inf_sum_list), THREADNUM)

  #Prepare plot.
  plt.plot(avg_vacc_list, label="vaccinated")
  plt.plot(avg_rec_list, label="recovered")
  plt.plot(avg_inf_list, label="infected")
  plt.title("Infected Population Simulation (sim = %d)\nRates: infection=%f, vaccination=%f, recovery: after %d days."%(THREADNUM,TAU,NU,MAX_DAYS))
  plt.xlabel("# days")
  plt.ylabel("# population")
  plt.legend(loc='upper left')
  plt.show()

  #Printing statistics about all threads run.
  print("========================================================")
  print("====================STATISTICS==========================")
  print("========================================================")
  print("  vacc_cnt: %d"%(vaccinated_cnt))
  print("  inf_cnt: %d"%(infected_cnt))
  print("  rec_cnt: %d"%(recovered_cnt))
  print("  vacc_sum_list: %s"%(str(vacc_sum_list)))
  print("  rec_sum_list: %s"%(str(rec_sum_list)))
  print("  inf_sum_list: %s"%(str(inf_sum_list)))
  print("  avg_vacc_list: %s"%(str(avg_vacc_list)))
  print("  avg_rec_list: %s"%(str(avg_rec_list)))
  print("  avg_inf_list: %s"%(str(avg_inf_list)))
else: #If we're just looking at one thread, we show the video.
  sim = Simulation(NROWS, NCOLS, TAU, NU, MAX_DAYS)
  run_sim_video(sim)
  # The total number of recovered patients at the end of the simulation instance. 
  sim.print_info()
