
#update - moved csp class to project interpreter
#from Finalclass import Csp
import sys
import time
from itertools import permutations
import queue #do not need
import numpy as np #do not need


#alphanumeric values for the dictionary to be created, letters are the rows, numbers are the columns.

rows = 'ABCDEFGHI'
columns = '123456789'
digits = columns

class Csp:

    #constructor for the csp class, our csp for the project will be a sudoku (9 by 9 grid with 81 total tiles)
    def __init__(self,grid):
        #initilize neighbors (the term peers is used in the book)
        self.peers = dict()
        self.domain = dict() #domain D
        self.trim = dict() #local search

        self.constraints = list() #initialize constraints created by the 3b3 units in create_constraints function
        
        
         #initialize local var.
        self.local_variables = list()  
         #constructs the grid for the excercise, finds the domain, variables with cross multiplication
        self.build(grid)
    
   
    
    
    def cross_multiplication(self,X,Y):

        '''Cross multiplication for rows and columns'''

        return [x+y for x in X for y in Y]
        
    def create_constraints(self):
        '''Create units'''

        #units are 3 by 3 tiles that contrain the sudoku for its completition, Fyi sudoku can't be solved if we have the same numbers in one of the units in the grid.
        #from norvig tutorial..
        
        units = ([self.cross_multiplication(rows,c) for c in digits] + [self.cross_multiplication(r,digits) for r in rows] + [self.cross_multiplication(r,c) for r in ('ABC','DEF','GHI') for c in ('123','456','789')])
        #check if you can create a func.

        #identify the subsets in the 3 by 3 units
        for unit in units:
            subsets = self.permutation(unit) #permutation function, will put below ()
            for subset in subsets:
                if [subset[0],subset[1]] not in self.constraints:
                    self.constraints.append([subset[0],subset[1]])
    

    def permutation(self,cycle):
        '''performs permutation formula on the unit subsets'''
        outcome = list()
        ranger = len(cycle) + 1
        for index in range(0,ranger):
            if index != 2:
                pass
            else:
                #returns permutation formula for the value inserted (itertools )
                for sub in permutations(cycle,index):
                    outcome.append(sub)
        
        return outcome
    

    def build_peers(self):
        '''build neighbors/peers'''

        for index in self.local_variables:
            self.peers[index] = list()
            for constraint in self.constraints:
                if index == constraint[0]:
                    self.peers[index].append(constraint[1])
    
    def build(self,grid):
        
        '''Builds the grid'''

        tiles = list(grid)
        self.local_variables = self.cross_multiplication(rows,columns)
        self.domain = {v: list(range(1,10)) if tiles[i] == '0' else [int(tiles[i])] for i,v in enumerate(self.local_variables)}
        
        self.trim = {v: list() if tiles[i] == '0' else [int(tiles[i])] for i,v in enumerate(self.local_variables)}
        self.create_constraints()
        self.build_peers()

    def is_solved(self):
        '''Checks if assignment has been solved'''

        for anyvar in self.local_variables:
            if len(self.domain[anyvar]) > 1:
                return False
            else:
                return True
        return True

    def is_complete(self,assignment):
        '''Checks if the assignment is complete'''

        for anyvar in self.local_variables:
            if len(self.domain[anyvar]) > 1 and anyvar not in assignment:
                return False
        
        return True

    def is_consistent(self,assignment,var, value): #

        ''' checks for consistency '''

        is_cons = True

        for key,val in assignment.items():
            if val == value:
                if key in self.peers[var]:
                    is_cons = False

        return is_cons

    def in_min_conflict(self,csp,var,val): #pg 221
        
        counters = 0
        
        for confliction in csp.peers[var]:
            if len(csp.domain[confliction]) > 1: 
                if val in csp.domain[confliction]:
                    counters = counters + 1
        
        return counters

    def constraint(self,i,j):
        '''Checks for constraints'''
        #returns a boolean
        return i != j

    def assign(self,variables,values,assingnment):
        '''assign and process forwardchecking'''

        assingnment[variables] = values
        
        self.forwardchecking(variables,values,assingnment)

    def forwardchecking(self,var,value,assignment):

        for peer in self.peers[var]:
            if peer not in assignment:
                if value in self.domain[peer]:
                    self.domain[peer].remove(value)
                    self.trim[var].append((peer,value))
    
    def unassign(self,var,assignment):
         
        if var in assignment:

            for (X,Y) in self.trim[var]: #binary components X D C
                self.domain[X].append(Y)

            self.trim[var] = []

            del assignment[var]

def selectunassinged(assignment,csp):
    '''pick the variable with the least legal moves possible'''

    unassigned = [v for v in csp.local_variables if v not in assignment]
    unassigned_variable = min(unassigned, key = lambda var: len(csp.domain[var]))
    
    return unassigned_variable

def val_domainorder(csp,var):
    
    domain_len = len(csp.domain[var])
    
    if domain_len == 1:
        
        return csp.domain[var]
    
    ordered = sorted(csp.domain[var], key = lambda val: csp.in_min_conflict(csp,var,val))
    
    return ordered

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def AC3(csp):
    '''AC3 Algorithm'''
    #return fals if an inconsistency is found, true otherwise
    
    arc_queue = list(csp.constraints) #Queue of arcs

    while arc_queue:

        Xi,Xj = arc_queue.pop(0) # pop from the first position of the queue

        if revise(csp,Xi,Xj): #if revise returns true

            if len(csp.domain[Xi]) == 0:
                return False
            
            for Xk in csp.peers[Xi]:
                if Xk != Xi:
                    arc_queue.append([Xk,Xi])
    return True
            
            #|
            #|
            #|
            #V

def revise(csp,Xi,Xj): # returns true if we revise the domain of Xi
    '''revise function'''
    is_revised = False

    for index in csp.domain[Xi]:

        if not any([csp.constraint(index,y) for y in csp.domain[Xj]]): #if no value of y allows (X,Y) to satisfy the constraint between xi and xj then delete x from Di
            
            csp.domain[Xi].remove(index)
            
            is_revised = True
    
    return is_revised

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def correct_input(csp):
    '''verifies if the length of the string is correct or not'''
    if len(csp) == 81: #did not use assert, bc vocareum should not have incorrect string inputs, it is mostly to check the flow of the program
        print('Correct length'.upper())
        print('pass'.upper())
        print('-----------------')
        print('\n')
    else:
        print('Wrong board length, excercise will not be correct'.upper())
        print('\n')

def BTS(assignment, csp):
    '''Backtracking search algorithm'''
    
    #a and b are assignment and variable placeholders, check they are equal in order to verify the sudoku completition
    a = len(assignment)
    b = len(csp.local_variables)
    
    if a == b: ##
        return assignment
    
    una_var = selectunassinged(assignment,csp) # --> select unassinged variable and store it in a variable for use

    for value in val_domainorder(csp,una_var):
        
        if csp.is_consistent(assignment,una_var,value): #if the value is consistent with assignment

            csp.assign(una_var,value,assignment)

            result = BTS(assignment,csp)

            if result:
                return result

            csp.unassign(una_var,assignment)
        
    return False


 

def main():
    algo = ''
    whitespace = ' '
    #project_sudoku = sys.argv[1]
    
    #jaja = '000260701680070090190004500820100040004602900050003028009300074040050036703018000'

    project_sudoku = sys.argv[1] #assign the string to a variable or use it in the CSP, doesn't matter.

    #jaja = ''
    csp = Csp(project_sudoku)
    print('Sudoku Board String : \n', project_sudoku)
    print('\n')
    print('-----------------')
    correct_input(project_sudoku) #using a function instead of assert method.
    
    #TRY THESE, ONE IS AC3
    #060000050902000061300006402400060000030052090500403000089000040000007200000820010
    #148697523372548961956321478567983214419276385823154796691432857735819642284765139
    #148697523372548961956321478567983214419276385823154796691432857735819642284765139
    #csp.display()
    
    if AC3(csp):
        print('AC3...')
    #if sudoku is ok with AC3, solve with AC3 FIRST
        if csp.is_solved():

            print('completed with ac3'.upper())
            print('-----------------')
            print('WRITING..........')
            print('\n')
            print('-----------------')
            algo = whitespace + 'AC3'
            file_out = open('output.txt','w')
            for line in csp.local_variables:
                file_out.write(str(csp.domain[line][0]))
            file_out.write(algo)           
            file_out.close()
            print('SUCCESS')
            print('-----------------')


        else:
            #use backtracking algorithm
            
            print('Could not complete with AC3, trying with BTS')
            print('BTS..')
            assignment = dict() #empty dic

            for index in csp.local_variables:
                if len(csp.domain[index]) == 1:
                    assignment[index] = csp.domain[index][0]
            
            assignment = BTS(assignment,csp)

            for d in csp.domain:
                csp.domain[d] = assignment[d] if len(d) > 1 else csp.domain[d]
                
            if assignment:

                print('completed with bts'.upper())
                print('-----------------')
                print('WRITING..........')
                print('\n')
                print('-----------------')
                algo = whitespace + 'BTS'
                file_out = open('output.txt','w')
                for line in csp.local_variables:
                    file_out.write(str(csp.domain[line]))
                file_out.write(algo)
                file_out.close()
                print('SUCCESS')
                print('-----------------')
            else:
                print('FAIL')
    
    print('Algorithm used:'.upper(),algo)
    print('-----------------')              

if __name__ == '__main__':
    
   
    start_time = time.time()

    main()

    end_time = time.time()
    
    print('Elapsed',end_time-start_time,' sec.')

    
    
