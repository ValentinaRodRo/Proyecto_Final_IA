from connect4 import Connect4
from agent_random import Agente_aleatorio
from agent_minimax import Agente_minimax

#crear agentes
minimax = Agente_minimax(jugadas_adelante=2)
random_agent = Agente_aleatorio()

def jugar_partida(agente1, agente2, mostrar=False):
    env = Connect4()
    agentes = {1: agente1, -1: agente2}

    while not env.final:
        agente = agentes[env.turno]
        accion = agente.elegir_accion(env)
        env.step(accion)

        if mostrar:
            print(f"\nTurno de {agente.nombre}, jugó columna {accion}")
            print(env.tablero)

    if mostrar:
        print(f"\nResultado final — Ganador: {env.ganador}")
    return env.ganador


#jugar varias partidas
n = 10
victorias = 0
empates = 0
for _ in range(n):
    ganador = jugar_partida(minimax, random_agent, mostrar=False)
    if ganador == 1:
        victorias += 1
    elif ganador == 0:
        empates += 1

print(f"De {n} partidas:")
print(f"Jugador 1 ganó {victorias}")
print(f"Empates: {empates}")
print(f"Jugador -1 ganó {n - victorias - empates}")
