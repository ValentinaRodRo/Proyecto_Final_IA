from connect4 import Connect4
from agent_random import Agente_aleatorio

env = Connect4()
agente = Agente_aleatorio()

for i in range(5):
    accion = agente.elegir_accion(env)
    print(f"Acción elegida: {accion}")
    env.step(accion)
    print(env.tablero)
