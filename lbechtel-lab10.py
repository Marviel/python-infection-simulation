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

NROWS = 100
NCOLS = 100
TIMESTEP = .1 #in seconds
MAX_DAYS = 20
TAU = .5
NU = .01






class Simulation(object):
  def __init__(self, nrows, ncols, tau, nu, max_days):
    self.nrows = nrows #cols
    self.ncols = ncols #rows
    self.tau = tau # Infection rate (between 0 and 1)
    self.nu = nu # Vaccination rate (between 0 and 1)
    self.k = max_days # Max days

    self.grid = [[0 for cols in range(self.ncols)] for rows in range(self.nrows)]

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
          if self.check_infect(r,c): 
            self.grid[r][c] = 1
          #If the patient has not become infected, the code should give him a chance to become vaccinated (using vaccination rate nu).
          if self.vaccinate(r,c): 
            self.grid[r][c] = -2
        elif self.grid[r][c] > 0:
          ninfected += 1 #increase infected count
          # If the patient is infected (matrix value is a positive integer), increment the integer once
          self.grid[r][c] += 1
          # check if it has exceeded k (the predetermined maximum number of days the disease can last).
          if self.grid[r][c] > self.k:
            self.grid[r][c] = -1 #If it has, set the value to -1 (recovered).

    return ninfected


  def check_infect(self, r, c):
  #def infect(Pop,i,j,n,m,tau):
    t=False

    # there should be a better way to calculate these, 
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

  def vaccinate(self, r, c):
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

  def infect_random(self):
    randr = random.randrange(self.nrows)
    randc = random.randrange(self.ncols)
    self.grid[randr][randc] = 1

  def print_grid(self):
    for row in range(self.nrows):
      print(self.grid[row])

  def get_rgb_color(self, val):
    if H[r][c] == -2:
      return [0,.6,0]
    elif H[r][c] == -1:
      return [0,.3,0]
    elif H[r][c] == 0:
      return [0,0,.5]
    else:
      norm = val/self.k
      return [norm,0,0]


class ThreadTest(Thread):

# Define any args you will be passing to the thread here:
  def __init__(self):
    Thread.__init__(self)

    sim = Simulation(NROWS, NCOLS, TAU, NU, MAX_DAYS)

    sim.infect_random()

  # The code that the thread executes goes here:
  def run(self):
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

for i in range(0,1) :
     current = ThreadTest(42,i)
     current.start()
     current.join()
     print ("\tThread ", i, " returnVal is: ", current.returnVal)



