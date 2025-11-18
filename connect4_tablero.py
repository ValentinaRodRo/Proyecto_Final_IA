import matplotlib.pyplot as plt
import numpy as np

class Connect4Tablero:
    def __init__(self, state, figsize=(6, 5)):
        self.state = state
        self.fig, self.ax = plt.subplots(figsize=figsize)
        plt.ion()
        self.draw()
    def draw(self):
        board = self.state.tablero
        filas, columnas = board.shape

        self.ax.clear()

        # Fondo azul del tablero
        for r in range(filas):
            for c in range(columnas):
                self.ax.add_patch(
                    plt.Rectangle((c, filas - 1 - r), 1, 1, color="blue")
                )

        # Fichas
        for r in range(filas):
            for c in range(columnas):
                val = board[r, c]
                if val == 1:
                    color = "yellow"
                elif val == -1:
                    color = "red"
                else:
                    color = "white"

                circle = plt.Circle(
                    (c + 0.5, filas - 1 - r + 0.5),
                    0.4,
                    color=color,
                    ec="black"
                )
                self.ax.add_patch(circle)

        self.ax.set_xlim(0, columnas)
        self.ax.set_ylim(0, filas)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        plt.pause(0.5)

    def update(self, new_state):
        self.state = new_state
        self.draw()