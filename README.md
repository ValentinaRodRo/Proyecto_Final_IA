DATOS NECESARIOS PARA LA EJECUCIÓN DEL AGENTE HELLO (GroupB)

Proyecto Final - Connect4

El GrupoB tiene una funcion que funciona como agente el cual se llama "Hello" el cual está diseñado para competir en el torneo definido anteriormente en donde se enfrentará contra otros agentes en partidas de Connect4

Hello implementa:
- Q-learning con memoria de estados
- Heurísticas de nivel experto (Victoria, Bloqueo)
- MCTS adaptativo por niveles de desempeño
- Entrenamiento progresivo

Todo esto le permite aprender, adaptarse y responder a otros agentes del torneo


El agente Hello opera bajo unos niveles, activando técnicas más avanzadas con el paso de su rendimiento

Dichos niveles son:
- NIVEL 0 - Q-Learning: Aqui el agente juega con explración Epsilon-greedy, aprende q-values para cada estado y actualiza su polítca cada turno

- NIVEL 1 - REGLAS DE VICTORIA: En este nivel el agente detecta jugadas ganadoras de inmediato y prioriza movimientos que hagan que el agente gane lo más rapido posible

- NIVEL 2 - REGLAS DE BLOQUEO: El agente en este nivel detecta jugadas ganadoras del oponentey trata de bloquearlas de inmediato

- NIVEL 3 - MCTS ADAPTATIVO. En este ultimo nivel, el agente es consistente y robusto y activa el MCTS, ademas de que el numero de simulaciones es dinámico segun la confianza en Q-table.



FUNCIONAMIENTO DEL HELLO EN EL TORNEO 
Durante el torneo, Hello se ejecuta en modo:
    Hello(modo_torneo=True)

Esto impliva que no se actualiza la q-table, no explora, usa unicamente su politica entrenada, aumenta la velocidad del agente y usa heuristicas+MCTS sin aprendizaje garantizando asi que el agente compite con politica fija y justa