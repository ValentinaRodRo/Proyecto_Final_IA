import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState


class Hello(Policy):

    
    #////inicialización del agente////
    
    #----constructor----
    def __init__(self, modo_torneo=False):
        self.modo_torneo = modo_torneo #cuando es True el agente no aprende

        #----Q-learning----
        self.Q = {} #Q-table: guarda cuanto vale cada acción en cada estado
        self.N = {} #contador de visitas: cuantas veces se ha tomado cada acción

        #----parametros de aprendizaje----
        self.alpha = 0.2  #tasa de aprendizaje (que tanto peso tiene la nueva información vs. la vieja)
        self.gamma = 1.0 #solo importa el resultado final

        #----memoria del turno anterior----
        self.estado_anterior = None
        self.accion_anterior = None

        #----epsilon dinamico----
        self.epsilon_inicial = 0.99#al inicio necesita explorar porque el agente no sabe nada
        self.epsilon_final = 0.05
        self.epsilon_decay = 0.995 #ya después no necesita explorar tanto
        self.epsilon = self.epsilon_inicial

        #----niveles de inteligencia----
        self.partidas_jugadas = 0
        self.nivel_actual = 0
        self.umbral_victoria = 20 
        self.umbral_bloqueo = 40
        self.umbral_mcts = 80

        #----generador aleatorio----
        self.rng = np.random.default_rng() #para elegir acciones aleatorias





    def mount(self):
        if self.modo_torneo:
            try:
                data = np.load("hello_model.npz", allow_pickle=True)
                self.Q = data["Q"].item()
                self.N = data["N"].item()
                self.nivel_actual = int(data["nivel"])
                self.epsilon = 0.0
            except:
                print("⚠ No se encontró modelo entrenado. Usando MCTS por defecto.")
                self.nivel_actual = 3   #activar MCTS aunque no haya entrenamiento
                self.epsilon = 0.0      #no explorar en torneo
                self.Q = {}             #Q vacío (no se usa si nivel 3)

        self.estado_anterior = None
        self.accion_anterior = None
    
    def actualizar_nivel(self):
        #el agente evoluciona por etapas según cuantas partidas haya jugado
        if self.partidas_jugadas >= self.umbral_mcts:
            self.nivel_actual = 3
        elif self.partidas_jugadas >= self.umbral_bloqueo:
            self.nivel_actual = 2
        elif self.partidas_jugadas >= self.umbral_victoria:
            self.nivel_actual = 1
        else:
            self.nivel_actual = 0


    #////funciones auxiliares////

    def clave_estado(self, tablero: np.ndarray):
        return tuple(map(tuple, tablero)) #convierte el tablero en una tupla de tuplas para usarse como clave en el diccionario self.Q

    #----verificar si un estado ya existe en Q-table----
    def verificar_estado(self, estado):
        if estado not in self.Q:
            self.Q[estado] = np.random.uniform(-0.2, 0.2, size=7) #Q-values en random
            self.N[estado] = np.zeros(7, dtype=int)

    def actualizar_Q(self, estado_ant, accion_ant, recompensa, estado_nuevo):
        self.verificar_estado(estado_ant)
        self.verificar_estado(estado_nuevo)

        Q_ant = self.Q[estado_ant][accion_ant] #mira el valor Q actual
        Q_obj = recompensa + self.gamma * np.max(self.Q[estado_nuevo]) #target: si después de colocar mi ficha sigo jugando bien, cual seria la mejor acción que puedo tomar en el futuro
        self.Q[estado_ant][accion_ant] += self.alpha * (Q_obj - Q_ant) #actualizamos el Q-value anterior hacia ese target
        self.N[estado_ant][accion_ant] += 1 


    
    #////detección de victoria inmediata o bloqueo////
    
    def accion_ganadora(self, state: ConnectState):
        
        #----revisa si podemos ganar con un solo movimiento----
        libres = list(state.get_free_cols())
        for col in libres:
            try:
                siguiente = state.transition(col) #para cada columna intenta simular poner una ficha ahí
                if siguiente.is_final() and siguiente.get_winner() == state.player:
                    return col
            except ValueError:
                continue
        return None

    def accion_bloqueo(self, state: ConnectState):
        
        #---revisa si el rival puede ganar en el proximo turno----
        rival = -state.player
        libres = list(state.get_free_cols())
        
        #simular que el rival juega en cada columna restante
        for col in libres:
            try:
                estado_rival = ConnectState(board=state.board.copy(), player=rival) #hacemos una copia del tablero y construimos un ConnectState donde el jugador activo es el rival
                siguiente = estado_rival.transition(col) #simulamos que el rival pone una ficha en col
                
                #---si el rival gana jugando en col, bloqueamos----
                if siguiente.is_final() and siguiente.get_winner() == rival:
                    return col
            except ValueError:
                continue
        return None


    #////MCTS con UCT////
    def mcts(self, estado_inicial: ConnectState, simulaciones=100, c=1.4):

        Qm = {} #suma de todas las recompensas obtenidas tomando la acción a en el estado s
        Nm = {} #veces que se ha elegido cada (estado, acción)
        Ns = {} #veces que hemos visitado cada estado (visitas totales a un nodo)

        def clave_nodo(st):
            return (tuple(map(tuple, st.board)), st.player) 

        #----formula UCT----
        def valor_uct(estado, accion):
            #decidir que acción explorar cuando está recorriendo el arbol
           
            if (estado, accion) not in Nm or Nm[(estado, accion)] == 0:
                return float('inf') #si no se ha simulado antes, forzamos la exploración
            
            #UCT = promedio de recompensa (explotación) + bono por explorar acciones poco probadas
            return (Qm[(estado, accion)] / Nm[(estado, accion)]) + c * np.sqrt(np.log(Ns[estado] + 1) / Nm[(estado, accion)])

        for _ in range(simulaciones):
            estado_actual = estado_inicial
            visitados = []

            #----SELECCIÓN + EXPANSIÓN----
            while True:
                clave_est = clave_nodo(estado_actual)
                libres = list(estado_actual.get_free_cols())

                if estado_actual.is_final() or len(libres) == 0:
                    break

                #1. si el nodo no se ha visto, expansión
                if clave_est not in Ns:
                    Ns[clave_est] = 0
                    for a in libres:
                        Nm[(clave_est, a)] = 0
                        Qm[(clave_est, a)] = 0.0
                    break #paramos la fase de selección porque solo se expande un nodo por simulación

                #2. el nodo ya existe, selección
                mejor_acc = max(libres, key=lambda a: valor_uct(clave_est, a)) #de todas las acciones posibles (columnas), elige la que tenga el mayor valor UCT
                visitados.append((clave_est, mejor_acc))

                try:
                    estado_actual = estado_actual.transition(mejor_acc) #simula colocar una ficha en mejor_acc, devolviendo un nuevo objeto ConnectState
                except ValueError:
                    break

            
            #----ROLLOUT----
            #simular una partida rapida desde un estado no expandido hasta llegar al final y ver que tan bueno es ese final para mi
            
            estado_sim = estado_actual #copia del estado donde termino la fase de selección/expansión
            iteraciones = 0
            while not estado_sim.is_final() and iteraciones < 200:
                iteraciones += 1

                libres = list(estado_sim.get_free_cols())
                if not libres:
                    break #si no hay columnas libres, se termina el rollout

                #política del rollout: preferir columnas centrales
                orden = [3, 2, 4, 1, 5, 0, 6]
                a = None
                for col in orden:
                    if col in libres:
                        a = col
                        break
                
                if a is None:
                    a = libres[0] #si ninguna columna del orden esta disponible se elige la primera columna legal

                try:
                    estado_sim = estado_sim.transition(a)#se coloca la ficha y se obtiene el nuevo estado
                except ValueError:
                    break

            #----fin del rollout: recompensa----
            ganador = estado_sim.get_winner()
            if ganador == estado_inicial.player:
                recompensa = 1
            elif ganador == -estado_inicial.player:
                recompensa = -1
            else:
                recompensa = 0

            
            #----BACKPROPAGATION----
            #le avisamos a todos los nodos del arbol que visitamos durante la simulación como nos fue
            
            for (est, a) in visitados:
                Nm[(est, a)] += 1 #aumenta el contador de veces que tomamos la acción a en el estado est
                Qm[(est, a)] += recompensa #suma la recompensa total acumulada de esa acción en ese estado
                Ns[est] += 1 #aumenta el numero de veces que visitamos ese estado

        #----valores finales----
        clave_raiz = clave_nodo(estado_inicial)
        libres = list(estado_inicial.get_free_cols())
        valores = np.zeros(7) #valores[0] = valor estimado de jugar en la columna 0...y así

        for a in libres:
            if (clave_raiz, a) in Nm and Nm[(clave_raiz, a)] > 0: #se probó esta columna a desde la raíz?
                valores[a] = Qm[(clave_raiz, a)] / Nm[(clave_raiz, a)] #promedio de recompensas obtenidas cuando MCTS eligió esa columna
            else:
                valores[a] = 0.0

        return valores


    #////política final////
    
    def act(self, board: np.ndarray) -> int:

        estado = self.clave_estado(board)
        self.verificar_estado(estado)

        #jugador actual
        num = np.count_nonzero(board)
        jugador = -1 if num % 2 == 0 else 1 #si hay  par le toca a -1 (rojo)

        state = ConnectState(board=board, player=jugador)
        libres = list(state.get_free_cols())

        # 1.Q-learning del turno anterior
        if (not self.modo_torneo) and self.estado_anterior is not None:  #mira a ver si hay un estado anterior para poder aprender
            ant_s = self.estado_anterior
            ant_a = self.accion_anterior

            #miramos cual fue la recompensa
            if state.is_final():
                w = state.get_winner()
                if w == jugador:
                    recompensa = -1
                elif w == -jugador:
                    recompensa = 1
                else:
                    recompensa = 0
            else:
                recompensa = 0

            self.actualizar_Q(ant_s, ant_a, recompensa, estado)

        # 2.si es terminal devuelve la primera columna libre para evitar errores
        if state.is_final():
            return libres[0]

        # 3.victoria inmediata
        if self.nivel_actual >= 1:
            acc = self.accion_ganadora(state)
            if acc is not None:
                self.estado_anterior = estado
                self.accion_anterior = acc
                return acc

        # 4.bloqueo inmediato
        if self.nivel_actual >= 2:
            acc = self.accion_bloqueo(state)
            if acc is not None:
                self.estado_anterior = estado
                self.accion_anterior = acc
                return acc

        # 5.MCTS
        valores_mcts = self.mcts(state, simulaciones=100)
        

        # 6.política epsilon-greedy
        if self.modo_torneo or self.nivel_actual < 3:
            valores = self.Q[estado].copy()
            for col in range(7):
                if col not in libres:
                    valores[col] = -999
        else:
            valores = self.mcts(state, simulaciones=120)

        # epsilon-greedy
        if (not self.modo_torneo) and self.rng.random() < self.epsilon:
            accion = int(self.rng.choice(libres))
        else:
            accion = int(max(libres, key=lambda a: valores[a]))

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

            if self.epsilon > self.epsilon_final:
                self.epsilon *= self.epsilon_decay

            self.partidas_jugadas += 1
            self.actualizar_nivel()
    
    def guardar_modelo(self, filename="hello_model.npz"):
        np.savez(
            filename,
            Q=self.Q,
            N=self.N,
            nivel=self.nivel_actual,
        )


