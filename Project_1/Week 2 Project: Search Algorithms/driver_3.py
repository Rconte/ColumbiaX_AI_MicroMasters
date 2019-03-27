
import sys
import math
import resource # ------> my IDE Will not load this, download or try it on class IDE **REMEMBER**
import time
import numpy as np





#skeleton code provided with puzzle movement and structure
### ++++ class skeleton code ++++++
    
class PuzzleState(object):

    
    def __init__(self, config, n, parent=None, action="Initial", cost=0):

        
        if n*n != len(config) or n < 2:
            

            raise Exception("the length of config is not correct!")
            

        self.n = n

        self.cost = cost

        self.parent = parent

        self.action = action

        self.dimension = n

        self.total_cost = -1

        self.config = config

        self.children = []

        for i, item in enumerate(self.config):

            if item == 0:

                self.blank_row = int(i / self.n)

                self.blank_col = i % self.n

                break

    
    def display(self):
        

        for i in range(self.n):

            line = []

            offset = i * self.n

            for j in range(self.n):

                line.append(self.config[offset + j])

            print(line)

    
    def move_left(self):
        

        if self.blank_col == 0:

            return None

        else:

            blank_index = int(self.blank_row * self.n + self.blank_col)

            target = blank_index - 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Left", cost=self.cost + 1)

    
    def move_right(self):
        

        if self.blank_col == self.n - 1:

            return None

        else:

            blank_index = int(self.blank_row * self.n + self.blank_col)

            target = blank_index + 1

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Right", cost=self.cost + 1)

    
    def move_up(self):
        

        if self.blank_row == 0:

            return None

        else:

            blank_index = int(self.blank_row * self.n + self.blank_col)

            target = blank_index - self.n

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Up", cost=self.cost + 1)

    
    def move_down(self):
        
        
        if self.blank_row == self.n - 1:

            return None

        else:

            blank_index = int(self.blank_row * self.n + self.blank_col)

            target = int(blank_index + self.n)

            new_config = list(self.config)

            new_config[blank_index], new_config[target] = new_config[target], new_config[blank_index]

            return PuzzleState(tuple(new_config), self.n, parent=self, action="Down", cost=self.cost + 1)

    def expand(self):
        

        # add child nodes in order of UDLR

        if len(self.children) == 0:
            

            up_child = self.move_up()

            
            if up_child is not None:
                

                self.children.append(up_child)

            down_child = self.move_down()
            

            if down_child is not None:
                

                self.children.append(down_child)

            left_child = self.move_left()
            

            if left_child is not None:
                

                self.children.append(left_child)

            right_child = self.move_right()

            
            if right_child is not None:
                

                self.children.append(right_child)
                

        return self.children
    
#2. Don't be satisfied by just using list as frontier. Instead, design your Frontier class which works faster.   
#using libraries such as deque or queue is totally fine, making a custom class gives the flexibility to use the same one for
# all search algorithms 
# in this case, the frontier class uses all the basic functions of a queue or dequeue library.    

class Frontier:

    '''CUSTOM FRONTIER CLASS'''

    #Constructor
    def __init__(self, initial_state):
        self.list = []
        self.set = set()

        self.list.append(initial_state)
        self.set.add(initial_state.config)
    
    #Enqueue the children(neighbors)
    def enqueue(self, state):
        
        self.list.insert(0,state)
        self.set.add(state.config)
    
    #Pushes the children(neighbors) onto the stack (more or less..)
    def push(self,state):
        
        self.list.append(state)
        self.set.add(state.config)
    
    #The U on the pseudocode should be implemented like this, sort of..
    def union(self,state):
        
        return state.config in self.set
    
    #Integrating the pop function within the custom class so it works with everything, we already have .pop() on python.
    def pop(self):
        
        state = self.list.pop()
        self.set.remove(state.config)
        return state
    
    #redo these -- are lambda functions just as effective as a normal function call? look this up..
    #START
    def lowPop(self):
        
        state = self.list.pop()
        self.set.remove(state.config)
        self.list.sort(key=lambda x: x.total_cost, reverse=True)
        return state

    def addW(self, state):
        
        self.list.append(state)
        self.set.add(state.config)
        self.list.sort(key=lambda x: x.total_cost, reverse=True)

    def updateW(self, state):
        
        for temp_state in self.list:
            if temp_state.config == state.config:
                temp_state.total_cost = state.total_cost
                break
        self.list.sort(key=lambda x: x.total_cost, reverse=True)  
    #END
    
    
    def len(self):
        
        return len(self.list)
    #Same function as a stack or queue class which can be found in countless textbooks, might aswell leave the method as shown on the         course pseudo..
    def isEmpty(self):
        
        return self.len() == 0


#The output file (example) will contain exactly the following lines:

#--------path_to_goal: ['Up', 'Left', 'Left']

#--------cost_of_path: 3

#--------nodes_expanded: 10

#--------search_depth: 3

#--------max_search_depth: 4

#--------running_time: 0.00188088

#--------max_ram_usage: 0.07812500

def write_output(state,nodes_expanded,max_depth):
    
    
    '''writes an output when we reach the goal'''

    path_to_goal = []
    temporary = state
    while temporary.parent != None:
        path_to_goal.insert(0, temporary.action)
        temporary = temporary.parent

    file = open("output.txt", "w")
    file.write("path_to_goal: {}\n".format(path_to_goal))
    file.write("cost_of_path: {}\n".format(state.cost))
    file.write("nodes_expanded: {}\n".format(nodes_expanded))
    file.write("search_depth: {}\n".format(state.cost))
    file.write("max_search_depth: {}\n".format(max_depth))
    file.write("running_time: {}\n".format(time.time() - time_start))
    #writing ram usage will not work on my IDE, the resource module is not on my system. But it is on the Voc. {resource library will not work on local machine, but works totally fine on vocareum}
    file.write("max_ram_usage: {}\n".format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    file.close()

    return


#ALGORITHMS
#1) First, we remove a node from the frontier set.
#2) Second, we check the state against the goal state to determine if a solution has been found.
#3) Finally, if the result of the check is negative, we then expand the node. To expand a given node, we generate successor nodes adjacent to the current node, 
#and add them to the frontier set. 
#Note that if these successor nodes are already in the frontier, or have already been visited, then they should not be added to the frontier again.

#Breadth first search algorithm -- used the pseudocode from the lectures or from Peter Norvig on Artificial Intelligence a Modern approach. 

def bfs_search(initial_state):
    
    #Initialize counters
    nodes_expanded = 0 
    max_depth = 0 
    #
    
    #initialize frontier with modified class and explored
    frontier = Frontier(initial_state) 
    explored = set()
    
    
    #Pseudocode in lectures, nothing more nothing less
    while not frontier.isEmpty():
        
        state = frontier.pop()  
        explored.add(state.config)

        
        if goal_test(state):
            
            write_output(state,nodes_expanded,max_depth) # ------> Reach the goal, Write!
            break

        if len(state.children) == 0:
            
            children = state.expand()
            nodes_expanded += 1
        
        for child in children:
            
            if frontier.union(child) or child.config in explored:
                continue
            
            if child.cost > max_depth:
                
                max_depth = child.cost
        
            frontier.enqueue(child) 



#Depth first search algorithm -- used the pseudocode from the lectures or from Peter Norvig on Artificial Intelligence a Modern approach. 

def dfs_search(initial_state): 
    
    #Initialize counters
    nodes_expanded = 0 
    max_depth = 0 
    #
    
    #initialize frontier with modified class and explored
    frontier = Frontier(initial_state)
    explored = set()

    
    #Pseudocode in lectures, nothing more nothing less
    while not frontier.isEmpty():
        
        state = frontier.pop()
        explored.add(state.config)

        if goal_test(state):
            
            write_output(state,nodes_expanded,max_depth) # ------> Reach the goal, Write!
            break

        if len(state.children) == 0:
            
            children = reversed(state.expand())
            nodes_expanded += 1

        for child in children:

            if frontier.union(child) or child.config in explored:
                
                continue

            if child.cost > max_depth:
                
                max_depth = child.cost

            frontier.push(child)

#A-Star search algorithm -- used the pseudocode from the lectures or from Peter Norvig on Artificial Intelligence a Modern approach. 

def A_star_search(initial_state):
    
    #Initialize counters
    nodes_expanded = 0 
    max_depth = 0 
    #
    
    #initialize frontier with modified class and explored
    frontier = Frontier(initial_state)
    explored = set()

    #Pseudocode in lectures, nothing more nothing less
    while not frontier.isEmpty():
        
        state = frontier.lowPop()
        explored.add(state.config)

        if goal_test(state):
            
            write_output(state,nodes_expanded,max_depth) # ------> Reach the goal, Write!
            break

        if len(state.children) == 0:
            
            children = reversed(state.expand())
            nodes_expanded += 1

        for child in children:
            
            if child.config in explored:
                
                continue
                
            child.total_cost = calculate_total_cost(child)

            if frontier.union(child):
                
                frontier.updateW(child)
                
                continue

            if child.cost > max_depth:
                
                max_depth = child.cost

            frontier.addW(child)
        

def goal_test(puzzle_state):

    return puzzle_state.config == (0, 1, 2, 3, 4, 5, 6, 7, 8)
#redo this



#calculate total cost of path
def calculate_total_cost(state):
    
        total = 0 
        
        for idx,square in enumerate(state.config):
            
            if square == 0:
                continue
               
            total += calculate_manhattan_dist_position(square,idx,state.n)
            
        return state.cost + total


def calculate_manhattan_dist_position(square,position,n):
    
    now_column = position % n
    now_row = int(position/n)
    
    goal_column = square %n
    goal_row = int(square/n)
    
    return abs(goal_column - now_column) + abs(goal_row- now_row)






#Storing the starting time in variable time_start
time_start = time.time()


def main():

    
    
    sm = sys.argv[1].lower()

    begin_state = sys.argv[2].split(",")

    begin_state = tuple(map(int, begin_state))

    size = int(math.sqrt(len(begin_state)))

    hard_state = PuzzleState(begin_state, size)

    #runs bfs,dfs,or astar algos
    if sm == "bfs":

        bfs_search(hard_state)
        #print(sm)

    elif sm == "dfs":

        dfs_search(hard_state)
        #print(sm)
    elif sm == "ast":

        A_star_search(hard_state)
        #print(sm)
    
    else:

        print("Enter valid command arguments !")

if __name__ == '__main__':
   
    main()
