def set_chess(board, x, y, color):
    board = [line.copy() for line in board]
    if board[x][y] != ' ':
        print('该位置已有棋子')
        print(x,y)
        for i in board:
            print(i)
        return None
    else:
        board[x][y] = color
        # for i in board:
        #     print(i)
        return board


def check_win(board):
    for list_str in board:
        if ''.join(list_str).find('O' * 5) != -1:
            # print('白棋获胜')
            return True
        elif ''.join(list_str).find('X' * 5) != -1:
            # print('黑棋获胜')
            return True
    else:
        return False


def check_win_all(board):
    board_c = [[] for line in range(29)]
    for x in range(15):
        for y in range(15):
            board_c[x + y].append(board[x][y])
    board_d = [[] for line in range(29)]
    for x in range(15):
        for y in range(15):
            board_d[x - y].append(board[x][y])
    return check_win(board) or check_win([list(l) for l in zip(*board)]) or check_win(board_c) or check_win(board_d)


def get_action(board, color=' '):
    actions = []
    for x in range(15):
        for y in range(15):
            if board[x][y] != color:
                for i in range(max(x - 1, 0), min(x + 2, 14)):
                    for j in range(max(y - 1, 0), min(y + 2, 14)):
                        if board[i][j] == color:
                            if (i, j) not in actions:
                                actions.append((i, j))
    return actions


class MinmaxAgent:
    def __init__(self):
        self.color = ['O', 'X']

    def alphabetaMax(self, gameState, depth, agentIndex, a, b):
        bestAction = None
        actions = get_action(gameState)
        if check_win_all(gameState):
            return -1, bestAction
        elif depth == 0 or len(actions) == 0:
            return 0, bestAction
        maxVal = -float('inf')
        for action in actions:
            gameState_ = set_chess(gameState, action[0], action[1], self.color[agentIndex])
            if gameState_ is None:
                continue
            value, _ = self.alphabetaMin(gameState_, depth - 1, (agentIndex + 1) % 2, a, b)
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
            if maxVal > a:
                a = maxVal
            if maxVal >= b:
                break
        return maxVal, bestAction

    def alphabetaMin(self, gameState, depth, agentIndex, a, b):
        bestAction = None
        actions = get_action(gameState)
        if check_win_all(gameState):
            return 1, bestAction
        elif depth == 0 or len(actions) == 0:
            return 0, bestAction
        minVal = float('inf')
        for action in actions:
            gameState_ = set_chess(gameState, action[0], action[1], self.color[agentIndex])
            if gameState_ is None:
                continue
            value, _ = self.alphabetaMax(gameState_, depth - 1, (agentIndex + 1) % 2, a, b)
            if value is not None and value < minVal:
                minVal = value
                bestAction = action
            if minVal < b:
                b = minVal
            if minVal <= a:
                break
        return minVal, bestAction

    def getMax(self, gameState, depth=0, agentIndex=0):
        bestAction = None
        actions = get_action(gameState)
        if check_win_all(gameState):
            return -1, bestAction
        elif depth == 0 or len(actions) == 0:
            return 0, bestAction
        maxVal = -float('inf')
        for action in actions:
            gameState_ = set_chess(gameState, action[0], action[1], self.color[agentIndex])
            if gameState_ is None:
                continue
            value, _ = self.getMin(gameState_, depth - 1, (agentIndex + 1) % 2)
            if value is not None and value > maxVal:
                maxVal = value
                bestAction = action
        return maxVal, bestAction

    def getMin(self, gameState, depth=0, agentIndex=1):
        bestAction = None
        actions = get_action(gameState)
        if check_win_all(gameState):
            return 1, bestAction
        elif depth == 0 or len(actions) == 0:
            return 0, bestAction
        minVal = float('inf')
        for action in actions:
            gameState_ = set_chess(gameState, action[0], action[1], self.color[agentIndex])
            if gameState_ is None:
                continue
            value, _ = self.getMax(gameState_, depth - 1, (agentIndex + 1) % 2)
            if value is not None and value < minVal:
                minVal = value
                bestAction = action
        return minVal, bestAction


import math
import random


class Node:
    def __init__(self, parent=None):
        self.child = {}
        self.parent = parent
        self.visit_times = 0
        self.qualuty_value = 0


class MCTS:
    def __init__(self):
        self.node = Node()
        self.color = ['O', 'X']
        self.agentIndex = 1

    def policy(self, gameState, agentIndex=0):
        agentIndex_ = agentIndex
        for i in range(2000):
            expand_node = self.node
            gameState_ = [line.copy() for line in gameState]
            while True:
                state = 0
                if check_win_all(gameState_):
                    break
                # expand
                actions = get_action(gameState_)
                for action in actions:
                    if action not in expand_node.child.keys():
                        gameState_ = set_chess(gameState_, action[0], action[1], self.color[agentIndex])
                        agentIndex = (agentIndex + 1) % 2
                        expand_node.child[action] = Node(expand_node)
                        expand_node = expand_node.child[action]

                        # Simulation
                        for i in range(10):
                            if check_win_all(gameState_):
                                break
                            actions_ = get_action(gameState_)
                            if len(actions_)==0:
                                break
                            action_ = random.choice(actions_)
                            gameState_ = set_chess(gameState_, action_[0], action_[1], self.color[agentIndex])
                            agentIndex = (agentIndex + 1) % 2
                        state = 1
                        break
                if state==1:
                    break

                # choose UCB
                score = -float('inf')
                action_ = list(expand_node.child.keys())[0]
                # print(expand_node.child.keys())
                expand_node_ = expand_node
                for action in expand_node.child.keys():
                    # C = 1 / math.sqrt(2)
                    C = 1.96
                    UCB = expand_node.child[action].qualuty_value / expand_node.child[action].visit_times + C * math.sqrt(
                        2 * math.log(expand_node.visit_times) / expand_node.child[action].visit_times)
                    if UCB > score:
                        expand_node_ = expand_node.child[action]
                        score = UCB
                        action_ = action
                expand_node = expand_node_
                gameState_ = set_chess(gameState_, action_[0], action_[1], self.color[agentIndex])
                agentIndex = (agentIndex + 1) % 2

            if check_win_all(gameState_):
                if (agentIndex + 1) % 2 == self.agentIndex:
                    reward = 1
                else:
                    reward = -1
            else:
                reward = 0

            # Back-propagation
            while expand_node != None:
                expand_node.visit_times += 1
                expand_node.qualuty_value += reward
                expand_node = expand_node.parent

            agentIndex = agentIndex_

        # return best
        score = 0
        action_ = list(self.node.child.keys())[0]
        for action in self.node.child.keys():
            UCB = self.node.child[action].qualuty_value / self.node.child[action].visit_times
            if UCB > score:
                score = UCB
                action_ = action
        return action_

    def action(self, action):
        # print(self.node.child)
        # for i in self.node.child.values():
        #     print(i.qualuty_value,i.visit_times)
        if action in self.node.child:
            self.node = self.node.child[action]
            self.node.parent = None

if __name__ == '__main__':
    ai = MinmaxAgent()
    mcts = None

    board = [[' '] * 15 for line in range(15)]

    for i in board:
        print(i)
    end = False
    for i in range(300):
        if i % 2 == 0:
            print('白棋下棋')
            while True:
                # x = input('请输入棋子坐标：')
                # chess = set_chess(board, int(x.split()[0]), int(x.split()[1]), 'O')

                if mcts == None:
                    chess = set_chess(board, 7, 7, 'O')
                else:
                    _,x = ai.alphabetaMin(board,4,0,-float('inf'),float('inf'))
                    chess = set_chess(board,x[0],x[1],'O')
                if chess is not None:
                    board = chess
                    for i in board:
                        print(i)
                    if check_win_all(board):
                        end = True
                        break
                    else:
                        break
                else:
                    continue
            if end:
                break
        else:
            print('黑棋下棋')
            while True:
                # x = input('请输入棋子坐标：')
                # chess = set_chess(board,int(x.split()[0]),int(x.split()[1]),'X')

                # # minmax
                # _, x = ai.alphabetaMin(board, 4, 1, -float('inf'), float('inf'))
                # chess = set_chess(board, x[0], x[1], 'X')

                # mcts
                mcts = MCTS()
                action = mcts.policy(board,1)
                mcts.action(action)
                chess = set_chess(board, action[0], action[1], 'X')

                if chess is not None:
                    board = chess
                    for i in board:
                        print(i)
                    if check_win_all(board):
                        end = True
                        break
                    else:
                        break
                else:
                    continue
            if end:
                break

# def alphbeta(depth,alpha,beta):
#     bestValue=-float('inf')
#     bestAction = None
#     actions = get_action(gameState)
#     if check_win_all(gameState):
#         return -1, bestAction
#     elif depth == 0 or len(actions) == 0:
#         return 0, bestAction
#
#     for action in actions:
#         gameState_ = set_chess(gameState, action[0], action[1], '')
#         if gameState_ is None:
#             continue
#         score,_=alphbeta(depth-1,-beta,-alpha)
#         score = -score
#         if (score>bestValue):
#               bestValue = score
#               bestAction=action
#             if (score >= alpha):
#                 alpha=score
#         if alpha >= beta:
#             break
#     return alpha,bestAction
#
# def MTDF(test,depth,alpha,beta):
#     while True:
#         bestValue=alphbeta(depth,test-1,test)
#         if bestValue<test:
#             test=beta=bestValue
#         else:
#             alpha=bestValue
#             test=bestValue+1
#         if alpha<beta:
#             break
#     return bestValue
#
# def deeping():
#     value = 0
#     depth = 1
#     while True:
#         value = MTDF(value,depth,-float('inf'),float('inf'))
#         depth+=1
#         if !time_out:
#             break
#     return value

# def PVS(depth,alpha,beta):
#     bestValue=-float('inf')
#     bestAction = None
#     actions = get_action(gameState)
#     if check_win_all(gameState):
#         return -1, bestAction
#     elif depth == 0 or len(actions) == 0:
#         return 0, bestAction
#
#     for action in actions:
#         gameState_ = set_chess(gameState, action[0], action[1], '')
#         if gameState_ is None:
#             continue
#         if bestValue==-float('inf'):
#             score, _ = alphbeta(depth - 1, -alpha - 1, -alpha)
#             score = -score
#             if (score > alpha and score < beta):
#                 score, _ = alphbeta(depth - 1, -beta, -alpha)
#                 score = -score
#         else:
#             score, _ = alphbeta(depth - 1, -beta, -alpha)
#             score = -score
#         if (score>bestValue):
#               bestValue = score
#               bestAction=action
    #         if (score >= alpha):
    #             alpha=score
#         if alpha >= beta:
#             break
#     return alpha,bestAction

# class ListNode:
#     def __init__(self,key,data):
#         self.key = key
#         self.value = data
#         self.next = None
# class HashMap:
#     def __init__(self,table_size):
#         self.items = [None]*table_size
#         self.count = 0
#
#     def __len__(self):
#         return self.count
#
#     def _hash(self,key):
#         return abs(hash(key))%len(self.items)
#
#     def __getitem__(self, item):
#         j = self._hash(item)
#         node = self.items[j]
#         while node is not None and node.key != item:
#             node = node.next
#         if node is None:
#             raise KeyError('KeyError'+repr(item))
#         return node
#
#     def __setitem__(self, key, value):
#         try:
#             node = self[key]
#             node.data = value
#         except KeyError:
#             j = self._hash(key)
#             node = self.items[j]
#             self.items[j]=ListNode(key,value)
#             self.items[j].next=node
#             self.count+=1
#
#     def __delitem__(self, key):
#         j = self._hash(key)
#         node = self.items[j]
#         if node is not None:
#             if node.key == key:
#                 self.items[j] = node.next
#                 self.count -= 1
#             else:
#                 while node.next != None:
#                     pre = node
#                     node = node.next
#                     if node.key == key:
#                         pre.next = node.next
#                         self.count -= 1
#                         break
#         else:
#             raise ValueError("元素不存在")
#
#     def __str__(self):
#         vals = []
#         for item in self.items:
#             temp_list = []
#             while item is not None:
#                 temp_list.append(str(item.key)+":"+str(item.data))
#                 item = item.next
#             vals.append("->".join(temp_list))
#         return str(vals)
#
#
# class Hashtable_Node:
#     def __init__(self):
#         self.lock = None
#         self.lower = None
#         self.upper = None
#         self.best_move = None
#         self.depth = None
#         self.next = None
# class Hashtable:
#     def __init__(self):
#         self.deepest = Hashtable_Node()
#         self.newest = Hashtable_Node()
# class HashMap_:
#     def __init__(self,Hashtable_size = 16,):
#         self.Hashtable_mask = Hashtable_size - 1
#         self.items = [Hashtable()]*Hashtable_size
#         self.count = 0
#
#         self.zobrist_swap_player = [0., 1.]
#         self.zobrist_black = [[random.random(), random.random()] for i in range(10)]
#         self.zobrist_white = [[random.random(), random.random()] for i in range(10)]
#         self.board = []
#         self.player = 0
#
#     def update(self,hashcode,lower,upper,best_move,depth):
#         node = self.items[hashcode[0]]
#         deepest = node.deepest
#         newest = node.newest
#         if hashcode[1]==deepest.lock and depth==deepest.depth:
#             if lower>deepest.lower:
#                 depth.lower = lower
#                 deepest.best_move = best_move
#             if upper<deepest.upper:
#                 deepest.upper = upper
#         elif hashcode[1]==newest.lock and depth==newest.depth:
#             if lower>newest.lower:
#                 depth.lower = lower
#                 newest.best_move = best_move
#             if upper<newest.upper:
#                 newest.upper = upper
#         elif depth>deepest.depth:
#             node.newest=deepest
#             deepest.lock=hashcode[1]
#             deepest.lower=lower
#             deepest.upper=upper
#             deepest.best_move=best_move
#             deepest.depth=depth
#         else:
#             newest.lock=hashcode[1]
#             newest.lower=lower
#             newest.upper=upper
#             newest.best_move=best_move
#             newest.depth=depth
#
#     def hash_get(self,hashcode,depth):
#         node = self.items[hashcode[0]]
#         deepest = node.deepest
#         newest = node.newest
#         if hashcode[1]==deepest.lock and depth==deepest.depth:
#             return deepest
#         elif hashcode[1]==newest.lock and depth==newest.depth:
#             return newest
#         else:
#             return None
#
#     def get_hashcode(self,hashcode):
#         hashcode[0] = 0
#         hashcode[1] = 0
#         for i in range(10):
#             if board[i] == 0:
#                 hashcode[0] += self.zobrist_black[i][0]
#                 hashcode[1] += self.zobrist_black[i][1]
#             elif board[i] == 1:
#                 hashcode[0] += self.zobrist_white[i][0]
#                 hashcode[1] += self.zobrist_white[i][1]
#         if self.player == 0:
#             hashcode[0] += self.zobrist_swap_player[0]
#             hashcode[1] += self.zobrist_swap_player[1]
#         return hashcode
#
#     def alpha_beta(self,alpha,beta,depth):
#         hashcode = []
#         hashcode = self.get_hashcode(hashcode)
#         node = self.hash_get(hashcode,depth)
#         if node:
#             if node.lower>alpha:
#                 alpha=node.lower
#                 if alpha>beta:
#                     return beta
#             if node.upper<beta:
#                 beta=node.upper
#                 if beta<=alpha:
#                     return beta
#         best_value,best_move = alphbeta(depth,alpha,beta)
#         if best_value>=beta:
#             self.update(hashcode,best_value,float('INF'),best_move,depth)
#         elif best_value<=alpha:
#             self.update(hashcode,best_value,-float('INF'),best_move,depth)
#         else:
#             self.update(hashcode, best_value, best_value, best_move, depth)
#         return best_value

# def Quies(alpha,beta):
#     val = e()
#     if val>=beta:
#         return beta
#     if val >alpha:
#         alpha=val
#     gen()
#     while(cap()):
#         nextcap()
#         val = -Quies(-beta,-alpha)
#         if val>=beta:
#             return beta
#         if val>alpha:
#             alpha=val
#     return alpha
#
# while time.time() - begin < 5:


