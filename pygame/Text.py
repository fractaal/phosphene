import pygame

class Text:
    def __init__(self, x, y, font_size=24, font_name="Input Mono", text=""):
        self.x = x
        self.y = y
        self.font = pygame.font.SysFont(font_name, font_size)
        self.text = text
        self.color = pygame.Color("white")

    def set_text(self, text):
        self.text = text

    def set_color(self, color):
        self.color = pygame.Color(color)

    def draw(self, screen):
        img = self.font.render(self.text, True, self.color)
        screen.blit(img, (self.x, self.y))
