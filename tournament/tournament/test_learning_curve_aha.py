import numpy as np
import matplotlib.pyplot as plt
import os
import importlib.util

# ============================================================
#  IMPORTAR Hello (GroupB) POR RUTA ABSOLUTA
# ============================================================

HELLO_PATH = os.path.join(
    os.path.dirname(__file__),
    "groups",
    "GroupB",
    "policy.py"
)

spec_h = importlib.util.spec_from_file_location("HelloModule", HELLO_PATH)
hello_module = importlib.util.module_from_spec(spec_h)
spec_h.loader.exec_module(hello_module)

Hello = hello_module.Hello


# ============================================================
#  IMPORTAR Aha (GroupA) POR RUTA ABSOLUTA
# ============================================================

AHA_PATH = os.path.join(
    os.path.dirname(__file__),
    "groups",
    "GroupA",
    "policy.py"
)

spec_a = importlib.util.spec_from_file_location("AhaModule", AHA_PATH)
aha_module = importlib.util.module_from_spec(spec_a)
spec_a.loader.exec_module(aha_module)

Aha = aha_module.Aha


# ============================================================
#  IMPORTS DEL ENTORNO
# ============================================================

from connect4.connect_state import ConnectState
from connect4.policy import Policy


# ============================================================
#  FUNCIÓN: JUGAR UNA PARTIDA Hello vs Aha
# ============================================================

def jugar_un_partido(agent1: Policy, agent2: Policy):
    """
    agent1 = Hello (siempre juega como rojo = -1)
    agent2 = Aha   (siempre juega amarillo = +1)
    
    Returns:
        -1 si gana Hello
        +1 si gana Aha
        0 si empate
    """
    board = np.zeros((6, 7), dtype=int)
    state = ConnectState(board=board, player=-1)  # Hello empieza

    while not state.is_final():
        if state.player == -1:
            col = agent1.act(state.board)
        else:
            col = agent2.act(state.board)

        try:
            state = state.transition(col)
        except ValueError:
            # Jugada ilegal => pierde automáticamente
            ganador = +1 if state.player == -1 else -1
            
            # Notificar a Hello del resultado
            if hasattr(agent1, 'finalizar_partida'):
                if ganador == -1:
                    agent1.finalizar_partida(1)   # Hello ganó
                elif ganador == 1:
                    agent1.finalizar_partida(-1)  # Hello perdió
                else:
                    agent1.finalizar_partida(0)   # Empate
            
            return ganador

    ganador = state.get_winner()
    
    # Notificar a Hello del resultado final
    if hasattr(agent1, 'finalizar_partida'):
        if ganador == -1:
            agent1.finalizar_partida(1)   # Hello ganó
        elif ganador == 1:
            agent1.finalizar_partida(-1)  # Hello perdió
        else:
            agent1.finalizar_partida(0)   # Empate
    
    return ganador


# ============================================================
#  FUNCIÓN: MEDIR LA CURVA DE APRENDIZAJE
# ============================================================

def medir_curva(n_partidas=200, batch=5):
    """
    Juega n_partidas entre Hello y Aha, midiendo:
    - Win rate en bloques de 'batch' partidas
    - Evolución de epsilon (exploración)
    - Cuándo se activa MCTS
    
    Args:
        n_partidas: Total de partidas a jugar
        batch: Tamaño del bloque para calcular win rate
    """
    
    print("="*60)
    print("INICIANDO EVALUACIÓN DE CURVA DE APRENDIZAJE")
    print("="*60)
    print(f"Total de partidas: {n_partidas}")
    print(f"Agente Hello: Q-learning + MCTS (con aprendizaje)")
    print(f"Agente Aha: Minimax (sin aprendizaje)")
    print("="*60)
    print()
    
    hello = Hello()
    aha = Aha(depth=3)

    resultados = []
    win_rates = []
    epsilons = []
    estados_q = []  # NUEVO: Para tracking de Q-table
    x_axis = []

    for i in range(n_partidas):
        r = jugar_un_partido(hello, aha)
        resultados.append(r)

        if (i + 1) % batch == 0:
            ultimos = resultados[-batch:]
            wins_hello = ultimos.count(-1)
            winrate = wins_hello / len(ultimos)
            win_rates.append(winrate)
            epsilons.append(hello.epsilon)
            estados_q.append(len(hello.Q))  # NUEVO: Guardar tamaño Q-table
            x_axis.append(i + 1)

            # Estado actual
            niveles = ["Q-learning", "Q + Victoria", "Q + Victoria + Bloqueo", "Q + Victoria + Bloqueo + MCTS"]
            nivel_nombre = niveles[hello.nivel_actual] if hello.nivel_actual < 4 else niveles[3]
            
            print(f"Partidas {i+1:3d}: WinRate={winrate:.2f} | "
                  f"Epsilon={hello.epsilon:.3f} | "
                  f"Nivel={hello.nivel_actual} ({nivel_nombre}) | "
                  f"Estados Q={len(hello.Q)}")

    print()
    print("="*60)
    print("ESTADÍSTICAS FINALES")
    print("="*60)
    
    total_wins_hello = resultados.count(-1)
    total_wins_aha = resultados.count(1)
    total_empates = resultados.count(0)
    
    print(f"Partidas totales: {n_partidas}")
    print(f"Victorias Hello: {total_wins_hello} ({total_wins_hello/n_partidas*100:.1f}%)")
    print(f"Victorias Aha: {total_wins_aha} ({total_wins_aha/n_partidas*100:.1f}%)")
    print(f"Empates: {total_empates} ({total_empates/n_partidas*100:.1f}%)")
    print(f"Epsilon final: {hello.epsilon:.4f}")
    print(f"Estados en Q-table: {len(hello.Q)}")
    
    niveles_texto = {
        0: "Q-learning puro",
        1: "Q + Victoria",
        2: "Q + Victoria + Bloqueo", 
        3: "Q + Victoria + Bloqueo + MCTS"
    }
    print(f"Nivel final: {hello.nivel_actual} ({niveles_texto.get(hello.nivel_actual, 'Desconocido')})")
    print("="*60)
    
    # ===== GRAFICAR =====
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))  # CAMBIADO: 3 gráficas
    
    # ----- Gráfica 1: Win Rate -----
    ax1.plot(x_axis, win_rates, marker='o', color='#2563eb', 
             linewidth=2, markersize=4, label="Win Rate Hello")
    ax1.axhline(y=0.5, color='red', linestyle='--', 
                linewidth=1.5, label="50% (equilibrio)")
    
    # Marcar cuando se activan habilidades
    if hello.nivel_actual >= 1 and hello.umbral_victoria <= n_partidas:
        ax1.axvline(x=hello.umbral_victoria, color='orange', linestyle=':', 
                    linewidth=1.5, alpha=0.6, label=f"Victoria activada ({hello.umbral_victoria})")
    
    if hello.nivel_actual >= 2 and hello.umbral_bloqueo <= n_partidas:
        ax1.axvline(x=hello.umbral_bloqueo, color='purple', linestyle=':', 
                    linewidth=1.5, alpha=0.6, label=f"Bloqueo activado ({hello.umbral_bloqueo})")
    
    if hello.nivel_actual >= 3 and hello.umbral_mcts <= n_partidas:
        ax1.axvline(x=hello.umbral_mcts, color='green', linestyle='--', 
                    linewidth=1.5, alpha=0.7, label=f"MCTS activado ({hello.umbral_mcts})")
    
    ax1.set_title("Curva de Aprendizaje - Hello (Q-learning + MCTS) vs Aha (Minimax)", 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel(f"Número de partidas (bloques de {batch})", fontsize=11)
    ax1.set_ylabel("Win Rate", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    # ----- Gráfica 2: Epsilon (Exploración) -----
    ax2.plot(x_axis, epsilons, marker='s', color='#16a34a', 
             linewidth=2, markersize=4, label="Epsilon (tasa de exploración)")
    ax2.axhline(y=hello.epsilon_final, color='orange', linestyle='--', 
                linewidth=1.5, label=f"Epsilon mínimo ({hello.epsilon_final})")
    
    if hello.nivel_actual >= 3 and hello.umbral_mcts <= n_partidas:
        ax2.axvline(x=hello.umbral_mcts, color='green', linestyle='--', 
                    linewidth=1.5, alpha=0.7, label=f"MCTS activado")
    
    ax2.set_title("Evolución de la Exploración", fontsize=14, fontweight='bold')
    ax2.set_xlabel(f"Número de partidas (bloques de {batch})", fontsize=11)
    ax2.set_ylabel("Epsilon", fontsize=11)
    ax2.set_ylim(0, max(epsilons) * 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    # ----- Gráfica 3: Crecimiento de Q-table -----
    ax3.plot(x_axis, estados_q, marker='D', color='#9333ea', 
             linewidth=2, markersize=4, label="Estados en Q-table")
    
    # Marcar hitos de aprendizaje
    if hello.umbral_victoria <= n_partidas:
        ax3.axvline(x=hello.umbral_victoria, color='orange', linestyle=':', 
                    linewidth=1.5, alpha=0.6, label=f"Victoria activada")
    
    if hello.umbral_bloqueo <= n_partidas:
        ax3.axvline(x=hello.umbral_bloqueo, color='purple', linestyle=':', 
                    linewidth=1.5, alpha=0.6, label=f"Bloqueo activado")
    
    if hello.umbral_mcts <= n_partidas:
        ax3.axvline(x=hello.umbral_mcts, color='green', linestyle='--', 
                    linewidth=1.5, alpha=0.7, label=f"MCTS activado")
    
    ax3.set_title("Crecimiento del Conocimiento", fontsize=14, fontweight='bold')
    ax3.set_xlabel(f"Número de partidas (bloques de {batch})", fontsize=11)
    ax3.set_ylabel("Estados Únicos en Q-table", fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='best')
    
    plt.tight_layout()
    plt.show()
    
    return hello, resultados, win_rates, epsilons, estados_q  # CAMBIADO: retornar estados_q


# ============================================================
#  ANÁLISIS ADICIONAL: Win Rate por Fases
# ============================================================

def analizar_fases(resultados, umbral_victoria=80, umbral_bloqueo=120, umbral_mcts=160):
    """
    Analiza el rendimiento en diferentes niveles de aprendizaje
    """
    
    nivel0 = resultados[:umbral_victoria]
    nivel1 = resultados[umbral_victoria:umbral_bloqueo]
    nivel2 = resultados[umbral_bloqueo:umbral_mcts]
    nivel3 = resultados[umbral_mcts:]
    
    print()
    print("="*60)
    print("ANÁLISIS POR NIVELES DE APRENDIZAJE")
    print("="*60)
    
    if len(nivel0) > 0:
        wins = nivel0.count(-1)
        total = len(nivel0)
        print(f"Nivel 0 - Q-learning puro (partidas 1-{total}):")
        print(f"  Win rate: {wins/total*100:.1f}%")
    
    if len(nivel1) > 0:
        wins = nivel1.count(-1)
        total = len(nivel1)
        print(f"Nivel 1 - Q + Victoria (partidas {umbral_victoria+1}-{umbral_victoria+total}):")
        print(f"  Win rate: {wins/total*100:.1f}%")
    
    if len(nivel2) > 0:
        wins = nivel2.count(-1)
        total = len(nivel2)
        print(f"Nivel 2 - Q + Victoria + Bloqueo (partidas {umbral_bloqueo+1}-{umbral_bloqueo+total}):")
        print(f"  Win rate: {wins/total*100:.1f}%")
    
    if len(nivel3) > 0:
        wins = nivel3.count(-1)
        total = len(nivel3)
        print(f"Nivel 3 - Todo + MCTS (partidas {umbral_mcts+1}-{len(resultados)}):")
        print(f"  Win rate: {wins/total*100:.1f}%")
    
    print("="*60)


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    # Ejecutar experimento
    hello, resultados, win_rates, epsilons = medir_curva(
        n_partidas=200,  # Puedes aumentar a 300 o 500
        batch=5
    )
    
    # Análisis adicional
    analizar_fases(resultados, umbral_mcts=hello.umbral_mcts)
    
    print("\n✅ Experimento completado exitosamente")