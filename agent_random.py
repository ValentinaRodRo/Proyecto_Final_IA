import numpy as np

class Agente_aleatorio: 
    #Agente que elige una acción váñida de manera aleatoria

    def __init__(self, nombre= "Agente Aleatorio", rng=None):

        self.nombre = nombre
        
        #se crea o usa un generador de números aleatorios
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng()


    def elegir_accion(self, env):
        #elige una acción random entre las columnas disponibles

        acciones = env.col_libre()

        return self.rng.choice(acciones)