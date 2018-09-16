import numpy as np
import random


def hashit(board):
    base3 = np.matmul(np.power(3, range(0, 9)), board.transpose())
    return int(base3)


def legal_moves(board):
    return np.where(board == 0)[0]


def epsilongreedy(board, player, epsilon, debug=False):
    moves = legal_moves(board)
    if (np.random.uniform() < epsilon):
        if (1 == debug):
            print("explorative move")
        return np.random.choice(moves, 1)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        va[i] = value[hashit(board)]
        board[moves[i]] = 0  # undo
    return moves[np.argmax(va)]


def iswin(board, m):
    if np.all(board[[0, 1, 2]] == m) | np.all(board[[3, 4, 5]] == m):
        return 1
    if np.all(board[[6, 7, 8]] == m) | np.all(board[[0, 3, 6]] == m):
        return 1
    if np.all(board[[1, 4, 7]] == m) | np.all(board[[2, 5, 8]] == m):
        return 1
    if np.all(board[[0, 4, 8]] == m) | np.all(board[[2, 4, 6]] == m):
        return 1
    return 0


def getotherplayer(player):
    if (player == 1):
        return 2
    return 1


def playRandom(board, player, games):
    games_left = np.size(legal_moves(np.zeros(9)))
    new_mean = 0
    value = 1
    for i in range(1, games + 1):
        games_left = np.size(legal_moves(board))
        temp_board = np.copy(board)
        temp_player = player
       
        for j in range(0, games_left):
            poss_moves = legal_moves(temp_board)
            action = random.choice(poss_moves)
            temp_board[action] = temp_player
            temp_player = getotherplayer(temp_player)
            if(iswin(temp_board, 2)):
                value = 1
                break
            elif(iswin(temp_board, 1)):
                value = 0
                break
            if(i==8):
                value = 0.5
                break

       


       
        old_mean = new_mean
        new_mean = old_mean + (1 / i) * (value - old_mean)
        #print(value)
        #print(new_mean)
        
        #printBoard(temp_board,0,3)
        #printBoard(temp_board,3,6)
        #printBoard(temp_board,6,9)

    return new_mean

def printBoard(board3,i,j):
    varr = [str(a) for a in board3[range(i,j)]]
    print(', '.join(varr))
    return 0
def compete(numgames,rollouts):
    game_result = np.zeros(numgames)
    for games in range(0, numgames):
        board = np.zeros(9)          # initialize the board
        sold = [0, hashit(board), 0]  # first element not used
        player = 1
        for move in range(0, 9):
            # use a policy to find action
            if(player == 2):
                action = play_mc(np.copy(board), player, rollouts)
            else:
                action = epsilongreedy(np.copy(board), player, 0.1)
            board[action] = player
            if (1 == iswin(board, player)):
                game_result[games] = player
                break
            if (8 == move):
                
                game_result[games] = 0.5  # draw (equal reward for both)
            player = getotherplayer(player)  # swap players
    return game_result

def learnit(numgames, epsilon, alpha, rollouts, debug=False):
    # play games for training
    for games in range(0, numgames):
        board = np.zeros(9)          # initialize the board
        sold = [0, hashit(board), 0]  # first element not used
        # player to start is "1" the other player is "2"
        player = 1
        # start turn playing game, maximum 9 moves
        
        res = []
        
        if((games % 1000) == 0):
            results2 = compete(100,rollouts)
            print(len(results2[results2==1.0]))
            res = [res,len(results2[results2==1.0])]
            
           

        for move in range(0, 9):
            # use a policy to find action
            action = epsilongreedy(np.copy(board), player, epsilon, debug)
            # perform move and update board (for other player)
            board[action] = player
            if debug:  # print the board, when in debug mode
                symbols = np.array([" ", "X", "O"])
                print("player ", symbols[player],
                      ", move number ", move + 1, ":")
                print(symbols[board.astype(int)].reshape(3, 3))

                # barr = [str(a) for a in board]
                # print(', '.join(barr))
                #print(hashit(board))
            if (1 == iswin(board, player)):  # has this player won?
                
                value[sold[player]] = value[sold[player]] +\
                    alpha * (1.0 - value[sold[player]])
                sold[player] = hashit(board)  # index to winning state
                value[sold[player]] = 1.0  # winner (reward one)
                value[sold[getotherplayer(player)]] =\
                    0.0  # looser (reward zero)
                break
            # do a temporal difference update,
            # once both players have made at least one move
            # setja in if player

            if (1 < move):
               
                value[sold[player]] = value[sold[player]] +\
                    alpha * (value[hashit(board)] - value[sold[player]])
               

            sold[player] = hashit(board)  # store this new state for player
            # check if we have a draw, then set
            # the final states for both players to 0.5
            if (8 == move):
                value[sold] = 0.5
                # draw (equal reward for both)
            player = getotherplayer(player)  # swap players
    return res

def play_mc(board, player,games):
    # possible afterstates(LEGALMOVES) after oldstate(sold)
    moves = legal_moves(board)
    na = np.size(moves)
    va = np.zeros(na)
    for i in range(0, na):
        board[moves[i]] = player
        va[i] = playRandom(board, 1, games)
        board[moves[i]] = 0  # undo

    return moves[np.argmax(va)]

    # iterate over afterstates and run random game to the end
    # after current state for each state.
    # record the winner and average over n = 10,30,100.



# global after-state value function, note
# this table is too big and contains states that
# will never be used, also each state is unique to the player
# (no one after-state seen by both players)

#def main(rollout):
    # for n = 10
    
    #compete(100, 0.1, 0.1, rollout)
    
    

value = np.ones(hashit(2 * np.ones(9))) / 1.0
data1 = learnit(10000, 0.1, 0.1,10)
print(data1)
value = np.ones(hashit(2 * np.ones(9))) / 1.0
data2 = learnit(10000, 0.1, 0.1,30)
print(data2)

value = np.ones(hashit(2 * np.ones(9))) / 1.0
data3 = learnit(10000, 0.1, 0.1,100)
print(data3)
# train the value function using 10000 games



# play one game deterministically using the value function
#learnit(1, 0, 0, True)
# play one game with explorative moves using the value function
#learnit(1, 0.1, 0, True)



#boardz = np.zeros(9)
#mean = playRandom(boardz, 1, 100)
#print(mean)

# varr = [str(a) for a in value]
# print(', '.join(varr))
