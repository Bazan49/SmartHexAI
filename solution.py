import time
import heapq
from board import HexBoard
import random
import math
from player import Player
from collections import defaultdict

class SmartPlayer(Player):
    
    def play(self, board: HexBoard) -> tuple:
        
        root = MCTS_Node(board.clone(), 3 - self.player_id, None, None)

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

        self.board = board # estado del tablero en este nodo
        self.parent = parent # nodo padre
        self.children = [] # nodos hijos

        # Estadísticas MCTS
        self.wins = 0
        self.visits = 0

        self.move = move  # Movimiento que llevó a este nodo (fila, col)
        self.player_id = player_id  # Jugador que puso la ultima ficha en ese tablero
        self.legal_moves = self.LegalMoves() # Movimientos legales desde este nodo
        self.untried_moves = self.legal_moves.copy() # Movimientos legales no explorados

        # Estadísticas RAVE
        self.rave_wins = defaultdict(int)
        self.rave_visits = defaultdict(int)

        
        self.my_cost = None # costo de mi camino ganador más corto
        self.opp_cost = None # costo de camino ganador más corto del oponente
        self.my_path = None # camino ganador más corto para mí
        self.opp_path = None # camino ganador más corto para el oponente

        self.compute_paths() 


    def compute_paths(self):

        """Calcula caminos ganadores más cortos para ambos jugadores"""

        size = self.board.size
        me = self.Next_Player()
        opp = self.player_id

        self.my_cost, self.my_path = self.GetShortestWinnerPath(self.board, size, me)
        self.opp_cost, self.opp_path = self.GetShortestWinnerPath(self.board, size, opp)


    @staticmethod
    def MCTS(root_state: MCTS_Node, time_limit: float, C: float = 1.41, k: int = 300):

        """Realiza el proceso MCTS desde el nodo raíz durante un tiempo limitado; devuelve el mejor movimiento encontrado"""

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

        """Un nodo está completamente expandido si no tiene movimientos legales sin explorar"""

        return len(self.untried_moves) == 0

    def Next_Player(self):
            
            """"Devuelve el id del siguiente jugador"""

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

        """Devuelve el hijo con mayor valor UCT"""

        best_score = -float("inf")
        best_node = None

        for child in self.children:
            
            score = child.CalculateUCT(C, k)

            if score > best_score:
                best_score = score
                best_node = child

        return best_node
    
    def expansion(self):

        """Expande el nodo eligiendo un movimiento no explorado con prioridad basada en caminos ganadores"""

         # Convertir los paths a sets para búsquedas rápidas
        my_path_set = set(self.my_path)
        opp_path_set = set(self.opp_path)

        # Generar lista de movimientos prioritarios
        priorities = []

        for move in self.untried_moves:
            r, c = move

            score = 0

            if move in my_path_set:
                score += 10  # favorece mi camino
            if move in opp_path_set:
                score += 5   # bloquea camino del rival
            if self.my_cost == 1 and move in my_path_set:
                score += 100  # ganar inmediatamente
            if self.opp_cost == 1 and move in opp_path_set:
                score += 50   # bloquear derrota inmediata

            priorities.append((score, move))

        #  Ordenar por prioridad
        priorities.sort(key=lambda x: x[0], reverse=True)

        #  Elegir primero de la lista ordenada
        _, move = priorities[0]

        #  Quitar de untried_moves y crear el nodo hijo
        self.untried_moves.remove(move)
        new_board = self.board.clone()
        next_player = self.Next_Player()
        new_board.place_piece(move[0], move[1], next_player)
        child = MCTS_Node(new_board, next_player, self, move)
        self.children.append(child)
        return child
    
    def simulation(self):

        """Realiza una simulación aleatoria desde el estado actual del nodo; devuelve el ganador y los movimientos jugados"""

        board = self.board.clone()
        player = self.Next_Player()
        
        # obtener movimientos legales
        moves = self.legal_moves.copy()

        played_moves = []

        while(moves):

            i = random.randrange(len(moves))
            move = moves[i]
            moves[i] = moves[-1]
            moves.pop()

            played_moves.append((move, player))

            board.place_piece(move[0], move[1], player)

            player = 3 - player 

        if(board.check_connection(3 - player)):
            return 3 - player, played_moves
        else:
            return player, played_moves


    def backpropagation(self, winner: int, played_moves = None):

        """Actualiza las estadísticas de este nodo y sus ancestros según el resultado de la simulación"""

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

        """Devuelve las coordenadas de los vecinos de una celda en el tablero hexagonal"""

        size = self.board.size

        if row % 2 == 0: #Fila par
            deltas = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]
        else:
            deltas = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]

        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                yield nr, nc

    
    # Para evaluar caminos ganadores más cortos, representamos el tablero como un grafo 
    # donde cada celda es un nodo y las aristas conectan celdas adyacentes. 
    # El costo de cada arista depende de si la celda está vacía (costo 1), ocupada por mí (costo 0) 
    # o por el oponente (costo infinito). Luego, aplicamos Dijkstra para encontrar el 
    # camino de costo mínimo desde mi lado del tablero hasta el lado opuesto, y lo mismo para el oponente.

    def GetShortestWinnerPath (self, board, size, player_id):

        """Devuelve el costo y camino ganador más corto para el jugador dado en el tablero"""

        adj, source, target = self.board_to_graph_with_sides(board, size, player_id)

        dist, parent = self.dijkstra(adj, source)

        cost = dist[target]

        path = self.get_path(parent, target, size)


        return cost, path
    
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

        # Min-heap (priority queue) para seleccionar el nodo con la distancia más corta no finalizado
        pq = []

        dist = [float('inf')] * V
        parent = [-1] * V

        # Inicializar la distancia del nodo fuente a 0 y agregarlo a la cola
        dist[src] = 0
        heapq.heappush(pq, (0, src))

        # Mientras haya nodos por explorar
        while pq:
            d, u = heapq.heappop(pq)

            # Si la distancia extraída es mayor que la distancia actual, ignorar
            if d > dist[u]:
                continue

            # Explorar vecinos de u
            for v, w in adj[u]:

                # Si se encuentra un camino más corto a v a través de u, actualizar la distancia y el padre
                if dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
                    parent[v] = u
                    heapq.heappush(pq, (dist[v], v))

        # Devolver las distancias y los padres para reconstruir caminos
        return dist, parent
    
    def get_path(self, parent, target, size):

        """Reconstruye el camino desde el nodo fuente hasta el nodo objetivo usando la lista de padres"""

        path = []
        cur = target

        while cur != -1:
            # ignorar source y target
            if cur < size * size:
                r, c = self.to_coord(cur, size)
                path.append((r,c))
            cur = parent[cur]

        path.reverse()
        return path
    