from connect4.policy import Policy
from connect4.connect_state import ConnectState
import numpy as np



class OhYes(Policy):

    # ---- constructor ----
    def __init__(self, modo_torneo=False):
        super().__init__()
        self.modo_torneo = modo_torneo 
        self.player = -1 
        
        # ---- Q-learning y Contadores ----
        self.Q = {} 
        self.N = {} 

        # ---- Parámetros de Aprendizaje ----
        self.alpha = 0.2 
        self.gamma = 1.0 
        self.epsilon_inicial = 0.99 
        self.epsilon_final = 0.10 # Mantenemos exploración
        self.epsilon_decay = 0.99 
        self.epsilon = self.epsilon_inicial 
        
        # ---- Historial de Q-learning ----
        self.estado_anterior = None
        self.accion_anterior = None

        # ---- Niveles de Inteligencia (Ajustados para 200 partidas) ----
        self.partidas_jugadas = 0
        self.nivel_actual = 0
        
        # Proporción para Q-learning, Victoria, y MCTS
        self.umbral_victoria = 70  # Comienza Q + Victoria
        self.umbral_mcts = 140     # Comienza Q + MCTS
        
        # ---- Parámetros MCTS (REDUCIDOS para velocidad) ----
        self.simulaciones_mcts = 100 # Reducidas de 240 a 100
        self.uct_c = 1.4

        # ---- Generador Aleatorio ----
        self.rng = np.random.default_rng(911)

    # ---- Montaje para Torneo ----
    def mount(self):
        # En el torneo, siempre se cargará el MCTS más alto (si existe).
        if self.modo_torneo:
            try:
                data = np.load("policyc_model.npz", allow_pickle=True)
                self.Q = data["Q"].item()
                self.nivel_actual = 2 # Nivel más alto
                self.epsilon = 0.0
            except:
                self.nivel_actual = 0 
                self.epsilon = 0.0  
                self.Q = {} 

        self.estado_anterior = None
        self.accion_anterior = None

    # ---- Actualización del Nivel (para la curva de aprendizaje) ----
    def actualizar_nivel(self):
        if self.partidas_jugadas >= self.umbral_mcts:
            self.nivel_actual = 2 # Q + MCTS (Máximo nivel)
        elif self.partidas_jugadas >= self.umbral_victoria:
            self.nivel_actual = 1 # Q + Victoria
        else:
            self.nivel_actual = 0 # Q-learning puro

    #//// funciones auxiliares ////

    def clave_estado(self, tablero: np.ndarray):
        return tuple(map(tuple, tablero)) 

    def verificar_estado(self, estado_clave):
        if estado_clave not in self.Q:
            self.Q[estado_clave] = np.array([np.random.uniform(-0.1, 0.1) for _ in range(7)])
            self.N[estado_clave] = np.zeros(7, dtype=int)

    def actualizar_Q(self, estado_ant, accion_ant, recompensa, estado_nuevo):
        # Lógica de actualización de Q-learning
        self.verificar_estado(estado_ant)
        self.verificar_estado(estado_nuevo)

        Q_ant = self.Q[estado_ant][accion_ant]
        # Búsqueda del máximo
        max_Q_futuro = -100.0
        for q_val in self.Q[estado_nuevo]:
            if q_val > max_Q_futuro:
                max_Q_futuro = q_val
                
        Q_obj = recompensa + self.gamma * max_Q_futuro 
        self.Q[estado_ant][accion_ant] += self.alpha * (Q_obj - Q_ant) 
        self.N[estado_ant][accion_ant] += 1

    #//// detección de victoria inmediata (Heurística) ////
    
    def accion_ganadora(self, state: ConnectState):
        libres = list(state.get_free_cols())
        for col in libres:
            try:
                siguiente = state.transition(col)
                if siguiente.is_final() and siguiente.get_winner() == state.player:
                    return col
            except ValueError:
                continue
        return None

    #//// MCTS con UCT ////
    
    def mcts(self, estado_inicial: ConnectState, simulaciones: int, c: float):
        # Implementación simple de MCTS 
        # Se cambia la distribucion en el numero de simulaciones para que el aprendizaje sea mayor

        Qm = {} 
        Nm = {} 
        Ns = {} 

        def clave_nodo(st):
            # Clave única (tablero, jugador)
            return (tuple(map(tuple, st.board)), st.player) 

        #Formula del UCT 
        def valor_uct(estado, accion):
            #En caso de que no sea visitado, se le da el valor de infinito para asi poder forzar la exploración
            if (estado, accion) not in Nm or Nm[(estado, accion)] == 0:
                return float('inf') 
            #Formula UCT: Explotacion (Q/N) + Exploracion (c*sqrt(log(Ns) / N))
            return (Qm[(estado, accion)] / Nm[(estado, accion)]) + c * np.sqrt(np.log(Ns[estado] + 1) / Nm[(estado, accion)])

        for _ in range(simulaciones):
            estado_actual = estado_inicial
            visitados = []
            
            # SELECCIÓN + EXPANSIÓN
            while True:
                clave_est = clave_nodo(estado_actual)
                libres = list(estado_actual.get_free_cols())

                if estado_actual.is_final() or not libres:
                    break
                # Expansion si el nodo es nuevo
                if clave_est not in Ns:
                    Ns[clave_est] = 0
                    for a in libres:
                        Nm[(clave_est, a)] = 0
                        Qm[(clave_est, a)] = 0.0
                    break 
                # Selección si el nodo ya existe
                mejor_acc = max(libres, key=lambda a: valor_uct(clave_est, a))
                visitados.append((clave_est, mejor_acc))
                
                try:
                    estado_actual = estado_actual.transition(mejor_acc)
                except ValueError:
                    break
            
            # ROLLOUT 
            estado_sim = estado_actual
            iteraciones = 0
            while not estado_sim.is_final() and iteraciones < 100: # Max 100 pasos en rollout
                iteraciones += 1
                libres = list(estado_sim.get_free_cols())
                if not libres: break
                # Politica del rollout: Eficiencia minima (elección aleatoria)
                a = libres[self.rng.choice(len(libres))] # Rollout aleatorio simple
                try:
                    estado_sim = estado_sim.transition(a)
                except ValueError:
                    break

            # Recompensa
            ganador = estado_sim.get_winner()
            #La recompensa en este caso llega a ser relativa al jugador que inició el MCTS
            recompensa = 1 if ganador == estado_inicial.player else -1 if ganador == -estado_inicial.player else 0

            # BACKPROPAGATION
            for (est, a) in visitados:
                Nm[(est, a)] += 1
                Qm[(est, a)] += recompensa
                Ns[est] += 1

        # RESULTADO (Explotación Pura)
        clave_raiz = clave_nodo(estado_inicial)
        libres = list(estado_inicial.get_free_cols())
        valores = np.full(7, -float('inf')) 
        
        for a in libres:
            if (clave_raiz, a) in Nm and Nm[(clave_raiz, a)] > 0: 
                #Promedio de recompensas (mejor estimación Q(s,a))
                valores[a] = Qm[(clave_raiz, a)] / Nm[(clave_raiz, a)] 
            else:
                valores[a] = 0.0 #Valor constante para acciones sin probar

        return valores


    #//// política final (El Cerebro) ////
    
    def act(self, board: np.ndarray) -> int:

        estado = self.clave_estado(board)
        self.verificar_estado(estado)
        libres = list(ConnectState(board=board, player=self.player).get_free_cols())

        # 1. Actualización de Q-learning del turno anterior
        if self.estado_anterior is not None:
            # Lógica de recompensa 
            current_state = ConnectState(board=board, player=self.player)
            w = current_state.get_winner()
            recompensa = -1 if w == self.player else 1 if w == -self.player else 0
            self.actualizar_Q(self.estado_anterior, self.accion_anterior, recompensa, estado)

        if not libres:
            return 3 # Columna central por defecto si no hay libres 

        # 2. HEURÍSTICA DE VICTORIA INMEDIATA 
        if self.nivel_actual >= 1:
            acc_ganadora = self.accion_ganadora(ConnectState(board=board, player=self.player))
            if acc_ganadora is not None:
                accion = acc_ganadora
                self.estado_anterior = estado
                self.accion_anterior = accion
                return accion

        # 3. MCTS o Q-learning
        
        if self.nivel_actual == 2:
            # MCTS (Nivel 2)
            state = ConnectState(board=board, player=self.player)
            valores_mcts = self.mcts(state, 
                                     simulaciones=self.simulaciones_mcts, 
                                     c=self.uct_c) 
            # Selecciona la mejor acción MCTS
            accion = int(max(libres, key=lambda a: valores_mcts[a]))
            
        else:
            # Q-learning (Nivel 0 y 1)
            
            # Epsilon-greedy (Exploración)
            if self.rng.random() < self.epsilon:
                # Exploración: Elegir al azar una columna libre
                accion = libres[self.rng.choice(len(libres))]
            else:
                # Explotación: Elegir el mejor Q-value
                q_valores = self.Q[estado]
                
                # Búsqueda de la mejor acción
                mejor_valor = -float('inf')
                mejor_accion = libres[0]
                for a in libres:
                    if q_valores[a] > mejor_valor:
                        mejor_valor = q_valores[a]
                        mejor_accion = a
                accion = mejor_accion


        # 4. Guardar estado y retornar
        self.estado_anterior = estado
        self.accion_anterior = accion
        return accion

    
    def finalizar_partida(self, recompensa_final):
        if not self.modo_torneo:
            if self.estado_anterior is not None:
                Q_ant = self.Q[self.estado_anterior][self.accion_anterior]
                self.Q[self.estado_anterior][self.accion_anterior] += self.alpha * (
                    recompensa_final - Q_ant
                )

            self.estado_anterior = None
            self.accion_anterior = None

            # Decaimiento del epsilon
            if self.epsilon > self.epsilon_final:
                self.epsilon *= self.epsilon_decay

            self.partidas_jugadas += 1
            self.actualizar_nivel()
    
    def guardar_modelo(self, filename="policyc_model.npz"):
        np.savez(
            filename,
            Q=self.Q,
            nivel=self.nivel_actual,
        )