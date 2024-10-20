import pygame
import math

class Spring:
    def __init__(self, position, target, stiffness=500, dampness=20, mass=5.0):
        self.position = pygame.Vector2(position)
        self.target = pygame.Vector2(target)
        self.velocity = pygame.Vector2(0, 0)
        self.stiffness = stiffness
        self.dampness = dampness
        self.mass = mass

    def update(self, dt):
        # Calculate the force using Hooke's law
        displacement = self.target - self.position
        spring_force = displacement * self.stiffness

        # Apply damping force
        damping_force = -self.velocity * self.dampness

        # Calculate acceleration (F = ma)
        acceleration = (spring_force + damping_force) / self.mass

        # Update velocity
        self.velocity += acceleration * dt

        # Update position
        self.position += self.velocity * dt

    def set_target(self, new_target):
        self.target = pygame.Vector2(new_target)

    def reset(self, position):
        self.position = pygame.Vector2(position)
        self.velocity = pygame.Vector2(0, 0)

    def get_position(self):
        return self.position
