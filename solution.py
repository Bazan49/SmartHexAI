import time
from hexboard import HexBoard
import random
import math
from player import Player
from collections import defaultdict

class SmartPlayer(Player):
    
    def play(self, board: HexBoard) -> tuple:
        
        root = MCTS_Node(board.clone(), 3 - self.player_id, None, None)
        best_child = MCTS_Node.MCTS(root, 5.0, 1.41, 300)

        print(f"Jugador {self.player_id} elige: {best_child.move} con {best_child.visits} visitas y {best_child.wins} victorias")
        return best_child.move

class MCTS_Node:
    def __init__(self, board: HexBoard, player_id: int, parent: MCTS_Node, move: tuple):
        self.board = board 
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0
        self.move = move  # Movimiento que llevó a este nodo (fila, col)
        self.player_id = player_id  # Jugador que puso la ultima ficha en ese tablero
        self.untried_moves = self.LegalMoves() # Movimientos legales no explorados
        self.rave_wins = defaultdict(int)
        self.rave_visits = defaultdict(int)

    @staticmethod
    def MCTS(root_state: MCTS_Node, time_limit: float, C: int = 1.41, k: int = 300):

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < time_limit:

            node = root_state

            # Selection
            
            while(node.is_fully_expanded() and node.children): # el nodo ya tiene todos los movimientos explorados
                node = node.best_child(C, k) # se selecciona el mejor hijo para seguir bajando por el árbol
            
            # Expansion
            if(not node.is_fully_expanded()): # el nodo tiene movimientos no explorados, se devuelve el nodo expandido a partir de él para simulación
                node = node.expansion()

            # 3. Simulation
            winner, played_moves = node.simulation()

            # 4. Backpropagation
            node.backpropagation(winner, played_moves)
            
        best_child = max(root_state.children, key=lambda n: n.visits)

        return best_child

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def Next_Player(self):
            return 3 - self.player_id # Cambia entre 1 y 2

    def LegalMoves(self):

        """Devuelve los movimientos legales"""

        moves = []
        for r in range(self.board.size):
            for c in range(self.board.size):
                if self.board.board[r][c] == 0:
                    moves.append((r, c))
        return moves
    
    def CalculateUCT(self, C, k):

        """Calcula el valor UTC de un nodo"""

        if(self.visits == 0):
            return float("inf")
        
        Beta = math.sqrt(k / (3 * self.visits + k))


        Q_Rave = self.parent.rave_wins[self.move] / self.parent.rave_visits[self.move] if self.parent.rave_visits[self.move] > 0 else 0
        Q_MCTS = self.wins / self.visits # Valor de explotación tradicional

        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)

        return Beta * Q_Rave + (1 - Beta) * Q_MCTS + C * exploration
    
    def best_child(self, C, k):
        best_score = -float("inf")
        best_node = None

        for child in self.children:
            
            score = child.CalculateUCT(C, k)

            if score > best_score:
                best_score = score
                best_node = child

        return best_node
    
    def expansion(self):

        i = random.randrange(len(self.untried_moves))
        move = self.untried_moves[i]
        self.untried_moves[i] = self.untried_moves[-1]
        move = self.untried_moves.pop() # Sacar un movimiento no explorado

        new_board = self.board.clone()
        next_player = self.Next_Player()
        new_board.place_piece(move[0], move[1], next_player)
        child = MCTS_Node(new_board, next_player, self, move)
        self.children.append(child)
        return child
    
    def simulation(self):

        board = self.board.clone()
        player = self.Next_Player()

        # obtener movimientos legales
        moves = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    moves.append((r, c))

        played_moves = []

        while(True):

            # verificar si alguien ganó (Tratar de optimizar esto)
            if board.check_connection(1):
                return 1 , played_moves
            if board.check_connection(2):
                return 2 , played_moves

            i = random.randrange(len(moves))
            move = moves[i]
            moves[i] = moves[-1]
            moves.pop()

            played_moves.append((move, player))

            board.place_piece(move[0], move[1], player)

            player = 3 - player # Cambiar de jugador (1 <-> 2)

    def backpropagation(self, winner: int, played_moves = None):

        node = self

        while node is not None:
            node.visits += 1
            if(node.player_id == winner):
                node.wins += 1

            # Actualizar estadísticas RAVE
            if played_moves is not None:
                for move, player in played_moves:
                    if(player == node.Next_Player()): #Actualizamos para el jugador que hizo el movimiento
                        node.rave_visits[move] += 1
                        if(player == winner):
                            node.rave_wins[move] += 1

            node = node.parent
    
   