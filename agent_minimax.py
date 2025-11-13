import numpy as np

class Agente_minimax:
    #agente Racional que toma decisiones usando busqueda limitada

    def __init__(self, jugadas_adelante = 3, nombre = "Agente Minimax"):
        
        self.jugadas_adelante = jugadas_adelante
        self.nombre = nombre

    def elegir_accion(self, env):
        #se elige la acción que maximiza la utilidad

        mejor_valor= -np.inf #por ahora el mejor valor es el numero más bajo que pueda existir
        mejor_acción = None

        for accion in env.col_libre():
            #se recorre todas las acciones posibles desde el estado actual

            estado_simulado = env.simular(accion)
            valor = self.min_valor(estado_simulado, self.jugadas_adelante-1) #cual jugada me perjudicaria más a mi

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_acción = accion
            
        return mejor_acción
    
    def max_valor(self, env, jugadas_adelante):
        #turno del jugador 1 (agente racional): intenta maximizar la utilidadd

        if env.final or jugadas_adelante == 0:
            return self.evaluar(env)
        
        valor = -np.inf

        for accion in env.col_libre():
            nuevo_estado= env.simular(accion)
            valor = max(valor, self.min_valor(nuevo_estado, jugadas_adelante-1))  #se evalua la respuesta del oponente (min_valor) y guarda el máximo
        return valor #se vuelve el mejor valor encontrado (la máxima utilidad para el agente)
    
    def min_valor(self, env, jugadas_adelante):
        #turno del jugador -1 (oponente), evalua el peor valor posible para el jugador 1

        if env.final or jugadas_adelante == 0:
            return self.evaluar(env)
        
        valor = np.inf #inicia con el mejor caso 

        for accion in env.col_libre():
            nuevo_estado = env.simular(accion)
            valor = min(valor, self.max_valor(nuevo_estado, jugadas_adelante -1)) # evalua el siguiente turno del agente

        return valor  #devuelve el peor valor (lo que el oponente elegiría para perjudicar al agente)
    
    def evaluar(self, env):

        if env.ganador == 1:
            return 1
        elif env.ganador == -1: 
            return -1
        else: 
            return 0
