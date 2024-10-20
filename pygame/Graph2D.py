import pygame
import numpy as np
from collections import deque

class Graph2D:
    def __init__(self, x, y, width, height, max_points=100):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.max_points = max_points
        self.data = deque(maxlen=max_points)
        self.x_min = float('-1')
        self.x_max = float('1')
        self.y_min = float('-1')
        self.y_max = float('1')

    def add_point(self, x_value, y_value):
        self.data.append((x_value, y_value))
        self.x_min = min(self.x_min, x_value)
        self.x_max = max(self.x_max, x_value)
        self.y_min = min(self.y_min, y_value)
        self.y_max = max(self.y_max, y_value)

    def draw(self, screen, color):
        if len(self.data) < 2:
            return

        # Draw border
        border_color = pygame.Color('gray')
        border_rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(screen, border_color, border_rect, 1)  # 1 is the border thickness

        x_scaled = np.interp([d[0] for d in self.data], (self.x_min, self.x_max), (0, self.width))
        y_scaled = np.interp([d[1] for d in self.data], (self.y_min, self.y_max), (self.height, 0))

        points = [(self.x + x, self.y + y) for x, y in zip(x_scaled, y_scaled)]

        # Draw lines with fading effect
        for i in range(1, len(points)):
            start = points[i-1]
            end = points[i]
            alpha = int(255 * (i / len(points)))

            # Handle both string color names and RGB tuples
            if isinstance(color, str):
                line_color = pygame.Color(color)
                line_color.a = alpha
            else:
                line_color = color + (alpha,)

            pygame.draw.line(screen, line_color, start, end, 2)

        # Draw axes
        axes_color = pygame.Color('darkgray')
        pygame.draw.line(screen, axes_color, (self.x, self.y + self.height), (self.x + self.width, self.y + self.height), 1)  # x-axis
        pygame.draw.line(screen, axes_color, (self.x, self.y), (self.x, self.y + self.height), 1)  # y-axis
