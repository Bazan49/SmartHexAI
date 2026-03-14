import time
from hexboard import HexBoard
import random
import math
from player import Player

class SmartPlayer(Player):
    
    def play(self, board: HexBoard) -> tuple:
        
        root = MCTS_Node(board.clone(), 3 - self.player_id, None, None)
        best_child = MCTS_Node.MCTS(root, 5.0)

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

    @staticmethod
    def MCTS(root_state: MCTS_Node, time_limit: float, C: int = 1.41):

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < time_limit:

            node = root_state

            # Selection
            
            while(node.is_fully_expanded() and node.children): # el nodo ya tiene todos los movimientos explorados
                node = node.best_child(C)
            
            # Expansion
            if(not node.is_fully_expanded()): # el nodo tiene movimientos no explorados, se devuelve el nodo expandido a partir de él para simulación
                node = node.expansion()

            # 3. Simulation
            winner = node.simulation()

            # 4. Backpropagation
            node.backpropagation(winner)
            
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
    
    def CalculateUCT(self, C):

        """Calcula el valor UTC de un nodo"""

        if(self.visits == 0):
            return float("inf")

        exploitation = self.wins / self.visits
        exploration = math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + C * exploration
    
    def best_child(self, C):
        best_score = -float("inf")
        best_node = None

        for child in self.children:
            
            score = child.CalculateUCT(C)

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

        while(True):

            # verificar si alguien ganó (Tratar de optimizar esto)
            if board.check_connection(1):
                return 1
            if board.check_connection(2):
                return 2

            i = random.randrange(len(moves))
            move = moves[i]
            moves[i] = moves[-1]
            moves.pop()

            board.place_piece(move[0], move[1], player)

            player = 3 - player # Cambiar de jugador (1 <-> 2)

    def backpropagation(self, winner: int):

        node = self

        while node is not None:
            node.visits += 1
            if(node.player_id == winner):
                node.wins += 1
            node = node.parent
    
   