import time
from player import Player
from board_Juan import HexBoard
import random
import math

class SmartPlayer(Player):
    
    def play(self, board: HexBoard) -> tuple:
        
        root = MCTS_Node(board.clone(), 3 - self.player_id, None, None)
        MCTS.PrincipalLoop(root, 5.0, 1.4)

        best_child = None
        best_visits = -1

        for child in root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

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

    def GetUntriedMoves(self):

        """Devuelve los movimientos legales no explorados"""

        legal_moves = self.LegalMoves()
        explored_moves = [child.move for child in self.children]
        
        untried_moves = []

        for move in legal_moves:
            if move not in explored_moves:
                untried_moves.append(move)
        
        return untried_moves
    
    def SelectRandomMove(self) -> tuple:

        """Selecciona un movimiento aleatorio entre los no explorados"""

        return random.choice(self.GetUntriedMoves())
    
   
class MCTS:

    @staticmethod
    def PrincipalLoop(root: MCTS_Node, time_limit: float, C: int):

        """Ejecuta el loop principal del MCTS durante un tiempo determinado"""

        start_time = time.perf_counter()

        while time.perf_counter() - start_time < time_limit:

            # SELECTION-EXPANSION
            selected_node = MCTS.selection(root, C)

            # SIMULATION
            result = MCTS.simulation(selected_node)

            # BACKPROPAGATION
            MCTS.backpropagation(selected_node, result)


    @staticmethod
    def CalculateUCT(node: MCTS_Node, C: int = 1.4):

        """Calcula el valor UTC de un nodo"""

        if(node.visits == 0):
            return float("inf")

        exploitation = node.wins / node.visits
        exploration = math.sqrt(math.log(node.parent.visits) / node.visits)

        return exploitation + C * exploration


    @staticmethod
    def selection(node: MCTS_Node, C: int):

        while(True):

            # si el nodo tiene movimientos no explorados, se devuelve el nodo expandido a partir de él para simulación
            untried_moves = node.GetUntriedMoves()
            if(len(untried_moves) != 0 ): 
                return MCTS.expansion(node, random.choice(untried_moves))
            
            # si no tiene hijos (estado terminal), se devuelve ese nodo para simulación
            if len(node.children) == 0:
                return node
            
            # Seleccionar próximo nodo a visitar
            next_node = None
            best_value = float("-inf")

            for child in node.children:

                uct_value = MCTS.CalculateUCT(child, C)

                if(uct_value > best_value):
                    best_value = uct_value
                    next_node = child

            node = next_node

    @staticmethod
    # Por ahora seleccionar un nodo random 
    def expansion(node: MCTS_Node, move: tuple):

        next_player = node.Next_Player()
        new_board = node.board.clone()
        new_board.place_piece(move[0], move[1], next_player)
        expanded_node = MCTS_Node(new_board, next_player, node, move)
        node.children.append(expanded_node)
        return expanded_node
    
    @staticmethod
    def simulation(node: MCTS_Node):

        board = node.board.clone()
        player = node.Next_Player()

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

    @staticmethod
    def backpropagation(node: MCTS_Node, result: int):

        while node is not None:
            node.visits += 1
            if(node.player_id == result):
                node.wins += 1
            node = node.parent
