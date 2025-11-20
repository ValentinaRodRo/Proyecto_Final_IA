import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Aha(Policy):

    def __init__(self, depth: int = 3):
        #constructor del agente
        self.depth = depth #cuantas jugadas hacia adelante mira el agente

    def mount(self):
        pass
    
    def act(self, board: np.ndarray) -> int:
        #mi jugada real, por cada jugada mia, pregunto: si yo hago esto que haria el rival después?

        #determinar el jugador que está jugando 
        num_piezas = np.count_nonzero(board)
        jugador_actual = -1 if num_piezas % 2 == 0 else 1 # si el número de piezas es par, juega -1 (rojo) y si es impar, juega +1 (amarillo)

        state = ConnectState(board=board, player=jugador_actual) #crea un estado a partir del tablero actual

        free_columns = list(state.get_free_cols())
        
        # Verificar que hay columnas libres
        if len(free_columns) == 0:
            # Si no hay columnas libres, devolver cualquier columna (esto no debería pasar)
            return 0

        #inicializa variables para encontrar la mejor jugada
        mejor_valor = -np.inf 
        mejor_columna = free_columns[0]

        for columna in free_columns:
            try:
                #crear un nuevo estado para cada evaluación
                state_para_evaluar = ConnectState(board=board, player=jugador_actual)
                estado_simulado = state_para_evaluar.transition(columna)
                valor = self.min_value(estado_simulado, self.depth - 1)

                #actualizar con la mejor jugada encontrada
                if valor > mejor_valor:
                    mejor_valor = valor
                    mejor_columna = columna
            except ValueError as e:
                #si hay un error con esta columna, la saltamos
                print(f"Error en columna {columna}: {e}")
                continue

        return mejor_columna
    
    #MINIMAX
    def max_value(self, state: ConnectState, depth: int) -> float:
        #turno del agente (maximiza la utilidad) osea mi respuesta a la respuesta del rival

        #Si el estado es final (gana, pierde o empate) devolvemos la utilidad del estado
        if state.is_final() or depth == 0: #si llegamos al limite de profundidad devolvemos también evaluate porque ya no exploramos más
            return self.evaluate(state)
        
        valor = -np.inf
        free_columns = list(state.get_free_cols())
        
        #si no hay columnas libres, es empate
        if len(free_columns) == 0:
            return 0

        for columna in free_columns:
            try:
                next_state = state.transition(columna) #simulamos la jugada sin alterar el estado original
                valor = max(valor, self.min_value(next_state, depth - 1))
            except ValueError:
                #saltar columnas invalidas
                continue
        
        return valor
    

    def min_value(self, state: ConnectState, depth: int) -> float:
        #turno del rival

        if state.is_final() or depth == 0:
            return self.evaluate(state)
        
        valor = np.inf #el rival quiere minimizar entonces usamos +inf como punto inicial
        free_columns = list(state.get_free_cols())
        
        #si no hay columnas libres, es empate
        if len(free_columns) == 0:
            return 0

        for columna in free_columns:
            try:
                next_state = state.transition(columna)
                valor = min(valor, self.max_value(next_state, depth - 1))
            except ValueError:
                # Saltar columnas inválidas
                continue
        
        return valor 
    
    #EVALUACIÓN

    def evaluate(self, state: ConnectState) -> float:
        ganador = state.get_winner()

        if ganador == state.player:
            return 1
        elif ganador == -state.player:
            return -1
        return 0