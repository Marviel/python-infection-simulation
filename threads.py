#!/usr/bin/env python3
from threading import Thread

class ThreadTest(Thread):

# Define any args you will be passing to the thread here:
     def __init__(self, n, k) :
           Thread.__init__(self)
           self.n = n
           self.k = k
           self.returnVal = -1

# The code that the thread executes goes here:
     def run(self) :
          n = self.n
          k = self.k

# You may define additional subroutines, for example:
          def subroutine1(x) :
              print ('Value {:2d} is printed by the subroutine in thread {:2d}'.format(x,k))

          subroutine1(n)
#         You will be able to access this value in the calling program
          self.returnVal = 86 + k

for i in range(0,11) :
     current = ThreadTest(42,i)
     current.start()
#    join() blocks the calling thread until the thread has terminated
#    comment out the join() call to see how many threads actually
#    complete before the the returnVal is printed
     current.join()
     print ("\tThread ", i, " returnVal is: ", current.returnVal)
