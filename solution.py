import time
import heapq
import sys
from numpy import size
from board import HexBoard
import random
import math
from player import Player
from collections import defaultdict

class SmartPlayer(Player):
    
    def play(self, board: HexBoard) -> tuple:
        
        root = MCTS_Node(board.clone(), 3 - self.player_id, None, None)
        root.compute_paths()

        # Si tengo un camino ganador de costo 1, juego en ese camino para ganar

        if(root.my_cost == 1):
             for (r, c) in root.my_path:
                if root.board.board[r][c] == 0:
                    return (r, c)
                
        # Si el oponente tiene un camino ganador de costo 1, juego en ese camino para bloquearlo

        if(root.opp_cost == 1):
            for (r, c) in root.opp_path:
                if root.board.board[r][c] == 0:
                    return (r, c)

        best_child = root.MCTS(root, 4.9, 1.0, 300)

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

        # Cosas de Dijkstra
        self.my_cost = None
        self.opp_cost = None
        self.my_path = None
        self.opp_path = None


    def compute_paths(self):

        size = self.board.size
        me = self.Next_Player()
        opp = self.player_id

        self.my_cost, self.my_path = self.GetShortestWinnerPath(self.board, size, me)
        self.opp_cost, self.opp_path = self.GetShortestWinnerPath(self.board, size, opp)


    @staticmethod
    def MCTS(root_state: MCTS_Node, time_limit: float, C: float = 1.41, k: int = 300):

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
        self.untried_moves.pop() # Sacar un movimiento no explorado

        new_board = self.board.clone()
        next_player = self.Next_Player()
        new_board.place_piece(move[0], move[1], next_player)
        child = MCTS_Node(new_board, next_player, self, move)
        self.children.append(child)
        return child
    
    def simulation(self):

        board = self.board.clone()
        player = self.Next_Player()
        last_move = None

        # obtener movimientos legales
        moves = []
        for r in range(board.size):
            for c in range(board.size):
                if board.board[r][c] == 0:
                    moves.append((r, c))

        played_moves = []

        while(moves):

            if(last_move is not None):

                move = self.check_real_bridge_threat(board, player, last_move)
                if(move is not None):
                    # Defender MI bridge atacado
                    moves.remove(move)

                else:
                    i = random.randrange(len(moves))
                    move = moves[i]
                    moves[i] = moves[-1]
                    moves.pop()

            else:
                i = random.randrange(len(moves))
                move = moves[i]
                moves[i] = moves[-1]
                moves.pop()

            played_moves.append((move, player))

            board.place_piece(move[0], move[1], player)

            last_move = move

            player = 3 - player # Cambiar de jugador (1 <-> 2)

        if(board.check_connection(3 - player)):
            return 3 - player, played_moves
        else:
            return player, played_moves


    def backpropagation(self, winner: int, played_moves = None):

        node = self

        while node is not None:
            node.visits += 1
            if(node.player_id == winner):
                node.wins += 1 # cuantas veces gano el jugador que hizo el movimiento que llevó a este nodo

            # Actualizar estadísticas RAVE
            if played_moves is not None:
                for move, player in played_moves:
                    if(player == node.Next_Player()): #Actualizamos para el jugador que hizo el movimiento
                        node.rave_visits[move] += 1
                        if(player == winner):
                            node.rave_wins[move] += 1

            node = node.parent

    def get_neighbors(self, row: int, col: int):
        if row % 2 == 0: #Fila par
            deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.board.size and 0 <= nc < self.board.size:
                yield nr, nc

    # MoHex utiliza un solo patrón durante la simulación de juegos aleatorios: 
    # si un jugador explora cualquier puente del oponente, entonces el oponente siempre 
    # responde de manera que mantenga la conexión. Si se exploran varios puentes simultáneamente,
    # entonces uno de esos puentes se selecciona al azar y luego se mantiene.

    def check_real_bridge_threat(self, board, player, candidate_move):
        """Detecta mis bridges amenazados por último movimiento del oponente"""
        opponent = 3 - player
        r, c = candidate_move
        
        # 1. Encontrar MIS piedras cerca del movimiento oponente
        my_stones_near = []
        for nr, nc in self.get_neighbors(r, c):  # Vecinos del oponente
            if (board.board[nr][nc] == player):  # MÍA
                my_stones_near.append((nr, nc))
        
        # 2. Si tengo ≥2 piedras mías cerca, buscar vecino_común VACÍO
        if len(my_stones_near) >= 2:
            for stone1 in my_stones_near:
                for stone2 in my_stones_near:
                    if stone1 != stone2:
                        # Vecinos compartidos entre mis dos piedras
                        neighbors1 = set(self.get_neighbors(*stone1))
                        neighbors2 = set(self.get_neighbors(*stone2))
                        common_neighbors = neighbors1 & neighbors2
                        
                        # 3. ¡BRIDGE ENCONTRADO! Vecino común VACÍO
                        for common_pos in common_neighbors:
                            cr, cc = common_pos
                            if (board.board[cr][cc] == 0):  # VACÍO
                                return (cr, cc)  # ¡DEFENDER AQUÍ!
        
        return None
    
    # Implementacion de caminos

    def GetShortestWinnerPath (self, board, size, player_id):

        adj, SOURCE, TARGET = self.board_to_graph_with_sides(board, size, player_id)

        dist, parent = self.dijkstra(adj, SOURCE)

        cost = dist[TARGET]

        path = self.get_path(parent, TARGET)

        real_path = []

        for node in path:
            if node < size * size:  # ignorar SOURCE y TARGET
                r, c = self.to_coord(node, size)
                real_path.append((r, c))

        return cost, real_path
    
    def to_coord(self, idx, size):
        return (idx // size, idx % size)
            
    def cell_cost(self, board, r, c, player_id):
        if board.board[r][c] == 0:
            return 1.0
        elif board.board[r][c] == player_id:
            return 0.0
        else:
            return float('inf')

    def board_to_graph_with_sides(self, board, size, player_id):
        V = size * size
        source = V  # nodo fuente adicional
        target = V + 1  # nodo objetivo adicional

        adj = [[] for _ in range(V + 2)] # lista de adyacencia 

        # 1. función coord → índice
        def idx(r, c):
            return r * size + c
        
        # 3. añadir aristas a vecinos
        for r in range(size):
            for c in range(size):

                for nr, nc in self.get_neighbors(r, c):
                        
                    w = self.cell_cost(board, nr, nc, player_id) # costo de ir a vecino

                    adj[idx(r, c)].append((idx(nr, nc), w)) # costo de ir desde (r,c) a (nr,nc)

         # --- conectar SOURCE y TARGET ---
        if player_id == 1:
            # izquierda → derecha
            for r in range(size):
                # izquierda
                w = self.cell_cost(board, r, 0, player_id)
                adj[source].append((idx(r, 0), w))

                # derecha
                w = self.cell_cost(board, r, size - 1, player_id)
                adj[idx(r, size - 1)].append((target, w))

        else:
            # arriba → abajo
            for c in range(size):
                # arriba
                w = self.cell_cost(board, 0, c, player_id)
                adj[source].append((idx(0, c), w))

                # abajo
                w = self.cell_cost(board, size - 1, c, player_id)
                adj[idx(size - 1, c)].append((target, w))

        return adj, source, target


    def dijkstra(self, adj, src):

        V = len(adj)

        # Min-heap (priority queue) storing pairs of (distance, node)
        pq = []

        dist = [sys.maxsize] * V
        parent = [-1] * V

        # Distance from source to itself is 0
        dist[src] = 0
        heapq.heappush(pq, (0, src))

        # Process the queue until all reachable vertices are finalized
        while pq:
            d, u = heapq.heappop(pq)

            # If this distance not the latest shortest one, skip it
            if d > dist[u]:
                continue

            # Explore all neighbors of the current vertex
            for v, w in adj[u]:

                # If we found a shorter path to v through u, update it
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))

        # Return the final shortest distances from the source
        return dist, parent
    
    def get_path(self, parent, target):
        path = []
        cur = target

        while cur != -1:
            path.append(cur)
            cur = parent[cur]

        path.reverse()
        return path
