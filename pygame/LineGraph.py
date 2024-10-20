import pygame
import numpy as np

class LineGraph:
    def __init__(self, x, y, width, height, max_points=100, color_gradient=True, color="white"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points
        self.data = []
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.color_gradient = color_gradient
        self.color=pygame.Color(color)

    def add_point(self, value):
        self.data.append(value)
        if len(self.data) > self.max_points:
            self.data.pop(0)
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    def draw(self, screen):
        if len(self.data) < 2:
            return

        scaled_data = np.interp(self.data, (self.min_value, self.max_value), (self.height, 0))

        if self.color_gradient:
            for i in range(len(self.data) - 1):
                start = (self.x + i * (self.width / (len(self.data) - 1)), self.y + scaled_data[i])
                end = (self.x + (i + 1) * (self.width / (len(self.data) - 1)), self.y + scaled_data[i + 1])

                # Calculate color based on value
                value_ratio = max((self.data[i] - self.min_value),0.01) / max((self.max_value - self.min_value),0.01)
                segment_color = self.interpolate_color(self.color, (255, 255, 255), value_ratio)

                pygame.draw.line(screen, segment_color, start, end, 2)
        else:
            points = [(self.x + i * (self.width / (len(self.data) - 1)), self.y + y)
                      for i, y in enumerate(scaled_data)]
            pygame.draw.lines(screen, self.color, False, points, 2)

    def interpolate_color(self, color1, color2, t):
        return tuple(int(a + (b - a) * t) for a, b in zip(color1, color2))
