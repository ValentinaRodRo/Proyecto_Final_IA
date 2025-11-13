import numpy as np

class Connect4:
    FILAS, COLUMNAS = 6, 7

    def __init__(self):
        #Tablero vacio
        self.tablero = np.zeros((self.FILAS, self.COLUMNAS), dtype =int)

        self.turno = 1 #1 es el turno rojo, -1 el turno amarillo
        self.final = False #Juego terminado (gana uno o empate)
        self.ganador = None

    def col_libre(self):
        return [col for col in range(self.COLUMNAS) if self.tablero[0, col]==0]
    
    def step(self, accion):
        #aplica una accion y devuelve el nuevo estado (funcion de transicion)

        fila = max(fil for fil in range (self.FILAS) if self.tablero[fil, accion]==0) #busca la ultima fila vacia en la columna que elige el jugador (accion)
        self.tablero[fila, accion] = self.turno

        if self.check_win (fila, accion):
            self.final = True
            self.ganador = self.turno 

        elif not self.col_libre():
            self.final=True
            self.ganador = 0 
        else:
            self.turno = self.turno * -1 
        return self.final, self.ganador
    
    def check_win (self, fila, col):
        jugador = self.tablero[fila, col]

        direcciones = [(1,0), (0,1), (1,1), (1, -1)]

        for df, dc in direcciones:
            contador = 1

            #hacia adelante
            f = fila + df
            c = col + dc

            while 0 <= f < self.FILAS and 0 <= c < self.COLUMNAS and self.tablero[f, c] == jugador:
                contador += 1
                f += df
                c += dc

            #hacia atras
            f = fila - df
            c = col - dc

            while 0 <= f < self.FILAS and 0 <= c < self.COLUMNAS and self.tablero[f, c] == jugador:
                contador += 1
                f-= df
                c -= dc

            if contador >= 4:
                return True
        
        return False
    
    def simular(self, accion):
        #creamos una copia del entorno para probar la acción que está pensado el agente para su siguiente jugada... para que el tablero original quede intacto
        
        nuevo = Connect4() #se crea un nuevo entorno vacío
        nuevo.tablero = self.tablero.copy() #copia el estado actual del tablero
        #copiamos todo lo demás
        nuevo.final = self.final
        nuevo.ganador = self.ganador

        nuevo.step(accion) #aplicamos la acción sobre el nuevo tablero
        
        return nuevo #nuevo pasa a representar el tablero después de hacer la jugada
    
    def reset(self):
        #reiniciamos el tablero a su estado inicial

        self.tablero[:] =0
        self.turno = 1
        self.final = False
        self.ganador = None
    
    


