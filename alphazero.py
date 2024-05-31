import copy
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def softmax(x):
    max = np.max(x, axis=0, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=0, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x

def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class policyNet(nn.Module):
    def __init__(self,mapsize):
        super(policyNet, self).__init__()

        self.mapsize = mapsize
        self.conv1 = nn.Conv2d(1,32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.act_conv1 = nn.Conv2d(128, 4, kernel_size=1)
        self.act_flatten = nn.Flatten()
        self.act_fc1 = nn.Linear(4*mapsize*mapsize,mapsize*mapsize)
        self.val_conv1 = nn.Conv2d(128,2,kernel_size=1)
        self.val_flatten = nn.Flatten()
        self.val_fc1 = nn.Linear(2*mapsize*mapsize,64)
        self.val_fc2 = nn.Linear(64,1)

    def forward(self,state):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        action = F.relu(self.act_conv1(x))
        action = F.log_softmax(self.act_fc1(self.act_flatten(action)),dim=1)

        value = F.relu(self.val_conv1(x))
        value = F.relu(self.val_fc1(self.val_flatten(value)))
        value = F.tanh(self.val_fc2(value))
        return action,value

class PilicyNet(nn.Module):
    def __init__(self,mapsize,model_file=None,use_gpu=False):
        super(PilicyNet, self).__init__()
        self.use_gpu = use_gpu
        self.mapsize = mapsize
        self.l2 = 1e-4
        if self.use_gpu:
            self.net = policyNet(mapsize).cuda()
        else:
            self.net = policyNet(mapsize)
        self.optim = optim.Adam(self.net.parameters(),weight_decay=self.l2)

        if model_file:
            self.net.load_state_dict(torch.load(model_file))

    def forward(self,state):
        state = torch.tensor(state,dtype=torch.float32)
        if self.use_gpu:
            state=state.cuda()
            log_act_probs,value = self.net(state)
            act_probs = np.exp(log_act_probs.cpu().numpy())
            return act_probs,value.cpu().numpy()
        else:
            log_act_probs,value = self.net(state)
            act_probs = np.exp(log_act_probs.detach().cpu().numpy())
            return act_probs,value.detach().numpy()

    def train_one_epoch(self,state,mcts_probs,winner_batch,lr):
        # print(state.shape)
        # state = np.concatenate(state,axis=0)
        # print(state)
        state = torch.tensor(state,dtype=torch.float32)
        # state = state[:,None,:,:]
        winner_batch = torch.tensor(winner_batch,dtype=torch.float32)
        winner_batch = torch.repeat_interleave(winner_batch,8,0)
        # mcts_probs = np.concatenate(mcts_probs,axis=0)
        mcts_probs = torch.tensor(mcts_probs,dtype=torch.float32)
        mcts_probs = F.softmax(mcts_probs,dim=1)
        mcts_probs = torch.repeat_interleave(mcts_probs,8,0)
        if self.use_gpu:
            state = state.cuda()
            mcts_probs = mcts_probs.cuda()
            winner_batch = winner_batch.cuda()

        self.optim.zero_grad()
        set_learning_rate(self.optim,lr)
        log_act_probs,value = self.net(state)
        print(value)
        print(winner_batch)
        print(mcts_probs)
        print(log_act_probs)
        value_loss = F.mse_loss(value.view(-1),winner_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs,1))
        loss = value_loss+policy_loss
        loss.backward()
        self.optim.step()
        entropy = -torch.mean(
            torch.sum(torch.exp(log_act_probs)*log_act_probs,1)
        )
        return loss,entropy

    def get_policy_param(self):
        return self.net.state_dict()

    def save_model(self,model_file):
        torch.save(self.get_policy_param(),model_file)

    def pre(self,state):
        state = np.concatenate(state,axis=0)
        state = torch.tensor(state)
        state = state[:,None,:,:]
        return state

class TreeNode:
    def __init__(self,parent):
        self.parent = parent
        self.child = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.s = 0

    def expand(self, action):
        self.child[action] = TreeNode(self)
        return self.child[action]

    def select(self, c_puct):
        return max(self.child.items(),key=lambda act_node:act_node[1].get_value(c_puct))

    def update(self,leaf_value):
        self.n_visits += 1
        self.s += leaf_value
        self.Q = (self.s)/self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self.u = (c_puct*np.sqrt(2*np.log(self.parent.n_visits)/(1+self.n_visits)))
        return self.Q+self.u

def rollout_policy(board):
    space = np.argwhere(board.map==0)
    action = space[np.argmax(np.random.rand(len(space)))]
    return action

def polict_value(board):
    space = np.argwhere(board.map == 0)
    return space

class MCTS:
    def __init__(self,c_puct=1.98,n_playout=1000):
        super(MCTS, self).__init__()

        self.root = TreeNode(None)
        self.c_puct = c_puct
        self.n_playout = n_playout

    def tree_policy(self,board):
        node = self.root
        choise = np.argwhere(board.map==0)
        player = 1
        end = False
        while len(choise)>0 and not end:
            choise = polict_value(board)
            if len(choise)==len(node.child):
                action, node = node.select(self.c_puct)
                player = (player + 1) % 2
                board.move(action[0], action[1], player + 1)
                end, winner = board.check(player+1)
            else:
                for i in choise:
                    action = (i[0],i[1])
                    if action not in node.child.keys():
                        node = node.expand(action)
                        player = (player + 1) % 2
                        board.move(action[0], action[1], player + 1)
                        end, winner = board.check(player+1)
                        break
                break
        return board,player,node

    def playout(self, board):
        board,player,node = self.tree_policy(board)
        leaf_value = self.evaluate_rollout(board,player)
        node.update_recursive(leaf_value)

    def evaluate_rollout(self,state,player,limit=100):
        winner=None
        for i in range(limit):
            end, winner = state.check(player+1)
            if end:
                break
            player = (player+1)%2
            action = rollout_policy(state)
            state.move(action[0],action[1],player+1)
        if winner==-1:
            return 0
        elif winner == 1:
                return -1
        elif winner == 2:
                return 1
        else:
            return 0


    def get_move(self,state):
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy)
        return max(self.root.child.items(),key=lambda act_node:act_node[1].Q)[0]

    def update_with_move(self,last_move):
        if (last_move[0],last_move[1]) in self.root.child:
            self.root = self.root.child[(last_move[0],last_move[1])]
            self.root.parent = None
        else:
            raise Exception('error')

    def __str__(self):
        return "MCTS"

class MCTSPlayer:
    def __init__(self,c_puct=5,n_playout=2000):
        self.mcts = MCTS(c_puct=5,n_playout=1000)

    def set_player_ind(self,p):
        self.player = p

    def reset_player(self,y,x):
        self.mcts.update_with_move((y,x))

    def get_action(self,board,player):
        if len(np.argwhere(board.map==0))>0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move((move[0],move[1]))
            return (move[0],move[1])

    def __str__(self):
        return "MCTS {}".format(self.player)

class Board(object):
    def __init__(self, **kwargs):
        self.mapsize = int(kwargs.get('mapsize', 8))
        self.n_in_row = int(kwargs.get('n_in_row', 5))
        self.players = [1, 2]


    def init_board(self):
        if self.mapsize < self.n_in_row:
            raise Exception('board width and height can not be '
                            'less than {}'.format(self.n_in_row))
        self.map = np.zeros((self.mapsize,self.mapsize))

    def move(self,y,x,player):
        if self.map[y,x]!=0:
            raise Exception('this place not empty ')
        self.map[y,x]=player

    def check(self, player):
        if len(np.argwhere(self.map == 0))==0:
            return True, -1
        index = np.argwhere(self.map == player)
        if len(index)<5:
            return False, player
        miny = np.min(index[:, 0])
        maxy = np.max(index[:, 0])
        minx = np.min(index[:, 1])
        maxx = np.max(index[:, 1])
        win_state = [player for i in range(self.n_in_row)]
        for i in range(miny, maxy+1):
            for j in range(minx, maxx+1):
                if j+self.n_in_row<=maxx+1:
                    line = self.map[i,j:j + self.n_in_row].tolist()
                    if line == win_state:
                        return True,player
                if i+self.n_in_row<=maxy+1:
                    line = self.map[i:i + self.n_in_row, j].tolist()
                    if line == win_state:
                        return True,player
                if j+self.n_in_row<=maxx+1 and i+self.n_in_row<=maxy+1:
                    line = self.map[range(i, i + self.n_in_row), range(j, j + self.n_in_row)].tolist()
                    if line == win_state:
                        return True,player
                    line = self.map[range(i, i + self.n_in_row), range(j + self.n_in_row - 1, j - 1, -1)].tolist()
                    if line == win_state:
                        return True,player
        return False,player

class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.mapsize
        height = board.mapsize

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(width):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                p = board.map[i,j]
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board()
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        current_player = start_player
        while True:
            player = self.board.players[current_player]
            current_player=1-current_player
            player_in_turn = players[player]
            y,x = player_in_turn.get_action(self.board,player)
            self.board.move(y,x,player)
            if player!=p2:
                player2.reset_player(y,x)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.check(player)
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players,winners_z = [], [], [],[]
        player_ = 1
        while True:
            move, move_probs = player.get_move_probs(self.board,player_)
            # store the data
            states.append(self.board.map)
            mcts_probs.append(move_probs)
            current_players.append(player_+1)
            # perform a move
            self.board.move(move[0],move[1],player_+1)
            if is_shown:
                self.graphic(self.board, p1, p2)
            end, winner = self.board.check(player_+1)
            player_ = 1-player_
            if end:
                winners_z = np.zeros(len(current_players))
                if winner != -1:
                    winners_z[np.array(current_players) == winner] = 1.0
                    winners_z[np.array(current_players) != winner] = -1.0
                # winner from the perspective of the current player of each state

                # reset MCTS root node
                player.reset_player()
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is player:", winner)
                    else:
                        print("Game end. Tie")
                return winner, zip(states, mcts_probs, winners_z)

class Human(object):
    """
    human player
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board, player):
        x, y = None,None
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            x,y = location
        except Exception as e:
            print(e)

        if x!=None:
            if int(board.map[x,y])==0:
                return x,y
        print("invalid move")
        y,x = self.get_action(board,player)
        return y, x

    def __str__(self):
        return "Human {}".format(self.player)

def run():
    n = 5
    width, height = 15, 15
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        mcts_player = MCTSPlayer(c_puct=5,
                                 n_playout=2000)  # set larger n_playout for better performance

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        # mcts_player = MCTS_Pure(c_puct=5, n_playout=1000)

        # human player, input your move in the format: 2,3
        human = Human()
        # human1 = Human()

        # set start_player=0 for human first
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


class MCTSP:
    def __init__(self,net,c_puct=1.98,n_playout=1000):
        super(MCTSP, self).__init__()

        self.root = TreeNodeP(None,1)
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.net = net

    def get_empty(self,board):
        indexs = np.argwhere(board==0).tolist()
        return indexs

    def tree_policy(self,board,player_):
        node = self.root
        player = player_
        end = False
        winner = 0
        while not end:
            if node.child=={}:
                break
            action, node = node.select(self.c_puct)
            board.move(action[0], action[1], player + 1)
            end, winner = board.check(player + 1)
            player = 1-player
        return board,player,node,end,winner

    def playout(self, board,player_):
        board,player,node,end,winner = self.tree_policy(board,player_)
        if not end:
            action_prob,_ = self.getvalue(board,node)
        index = self.get_empty(board.map)
        while len(index)>0:
            i = random.choice(index)
            board.move(i[0], i[1], player + 1)
            end, winner = board.check(player + 1)
            player = 1 - player
            index.remove(i)
            if end:
                break

        if winner == -1:
            leaf_value = 0
        elif winner == 1:
            leaf_value = -1
        elif winner == 2:
            leaf_value = 1
        else:
            leaf_value = 0

        node.update_recursive(leaf_value)


    def getvalue(self,board,node):
        action_prob,value = self.net(board.map.reshape(1,1,board.map.shape[0],board.map.shape[1]))
        for i in self.get_empty(board.map):
            node.expand((i[0],i[1]),action_prob)
        return action_prob,value[0,0]

    def get_move(self,state,player):
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy,player)
        return max(self.root.child.items(),key=lambda act_node:act_node[1].Q)[0]

    def get_move_probs(self,state,player):
        for n in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self.playout(state_copy,player)
        act_visits = [(act, node.Q)
                      for act, node in self.root.child.items()]
        acts, act_probs = zip(*act_visits)
        move = acts[np.random.choice(
            range(len(acts)),
            p=softmax(0.75 * np.array(act_probs) + 0.25 * np.random.dirichlet(0.3 * np.ones(len(act_probs))))
        )]
        self.update_with_move(move)
        probs = np.zeros(state.map.shape)
        for a in act_visits:
            probs[a[0]]=a[1]
        probs = probs.reshape(-1)
        return move,probs

    def update_with_move(self,last_move):
        if (last_move[0],last_move[1]) in self.root.child:
            self.root = self.root.child[(last_move[0],last_move[1])]
            self.root.parent = None
        else:
            raise Exception('error')

    def reset_player(self):
        self.root = TreeNodeP(None,1)

    def __str__(self):
        return "MCTS"

class TreeNodeP:
    def __init__(self,parent,P):
        self.parent = parent
        self.child = {}
        self.n_visits = 0
        self.Q = 0
        self.u = 0
        self.s = 0
        self.P = P

    def expand(self, action,action_prob):
        self.child[action] = TreeNodeP(self,action_prob[0,action[0]*8+action[1]])
        return self.child[action]

    def select(self, c_puct):
        return max(self.child.items(),key=lambda act_node:act_node[1].get_value(c_puct))

    def update(self,leaf_value):
        self.n_visits += 1
        self.s += leaf_value
        self.Q = (self.s)/self.n_visits

    def update_recursive(self, leaf_value):
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self.u = (c_puct*self.P*np.sqrt(self.parent.n_visits)/(1+self.n_visits))
        return self.Q+self.u

from collections import deque
class train:
    def __init__(self):
        self.board = Board(width=15, height=15, n_in_row=5)
        self.game = Game(self.board)
        self.lr = 2e-3
        self.lr_mult = 1
        self.playout = 400
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 512
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5
        self.kl_targ = 0.02
        self.check_freq = 10
        self.game_batch_num = 1500
        self.best_win_atio = 0
        self.net = PilicyNet(8)
        self.mcts_player = MCTSP(self.net)

    def get_equi_data(self, data_):
        data = []
        for board, mcts_probs, winners_z in data_:
            boards =[]
            boards.append(board)
            boards.append(board.T)
            boards.append(np.flip(board))
            boards.append(np.flip(board.T))
            boards.append(np.flip(board,0))
            boards.append(np.flip(board,1))
            boards.append(np.flip(board.T,0))
            boards.append(np.flip(board.T,1))
            data.append([boards,mcts_probs, winners_z])
        return data

    def collect_data(self,e=1):
        for i in range(e):
            winner,data = self.game.start_self_play(self.mcts_player)
            data = list(data)[:]
            self.len = len(data)
            data = self.get_equi_data(data)
            self.data_buffer.extend(data)

    def policy_update(self):
        mini_batch = random.sample(self.data_buffer,self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        prob_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        state_batch = self.net.pre(state_batch)
        old_probs, old_v = self.net(state_batch)
        for i in range(self.epochs):
            loss,entropy = self.net.train_one_epoch(state_batch,prob_batch,winner_batch,self.lr*self.lr_mult)
            new_probs, new_v = self.net(state_batch)
            kl = np.mean(np.sum(old_probs*(np.log(old_probs+1e-10)-np.log(new_probs+1e-10)),axis=1))
            if kl>self.kl_targ:
                break
        if kl > self.kl_targ*2 and self.lr_mult>0.1:
            self.lr_mult /= 1.5
        elif kl<self.kl_targ/2 and self.lr_mult<10:
            self.lr_mult *= 1.5
        return loss,entropy,kl

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                self.collect_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(
                        i+1, self.epochs))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy, kl = self.policy_update()
                    print("loss:{}, entropy:{},kl:{}".format(
                        loss, entropy, kl))
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.net.save_model('./current_policy.model')
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    # run()
    t = train()
    t.run()