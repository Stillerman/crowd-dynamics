"""This example spawns (bouncing) balls randomly on a L-shape constructed of 
two segment shapes. Not interactive.
"""

__version__ = "$Id:$"
__docformat__ = "reStructuredText"

# Python imports
import random
from typing import List
import matplotlib.pyplot as plt

# Library imports
import pygame

# pymunk imports
import pymunk
import pymunk.pygame_util
import numpy as np


class CrowdSim(object):

    def __init__(self, MAX_VEL=100) -> None:

        # Params
        self.MAX_VEL = MAX_VEL
        self.WALL_LENGTH = 500
        self.OFFSET = 10
        self.EXIT = (self.OFFSET + self.WALL_LENGTH / 2,
                     self.OFFSET + 0.95 * self.WALL_LENGTH)

        self.history = []

        self._init_world()

        self.init_walls()
        self.init_people()


    def init_walls(self) -> None:
        static_body = self._space.static_body

        # Create segments around the edge of the screen.
        l = self.WALL_LENGTH
        offset = self.OFFSET

        walls = [
            pymunk.Segment(static_body, (offset, offset), (l, offset), 0.0),
            pymunk.Segment(static_body, (l, offset), (l, l), 0.0),
            pymunk.Segment(static_body, (l, l), (offset, l), 0.0),
            pymunk.Segment(static_body, (offset, l), (offset, offset), 0.0),
        ]

        for wall in walls:
            wall.elasticity = 0.95
            wall.friction = 0.9
        self._space.add(*walls)

    def update_people(self) -> None:

        # make people want to exit
        for person,speed in self.people:
            towards_exit = (np.asarray(self.EXIT) -
                            np.asarray(person.body.position))
            towards_exit = towards_exit / np.linalg.norm(towards_exit)
            towards_exit = towards_exit * 500
            person.body._set_force(tuple(towards_exit))

            velocity = person.body._get_velocity()
            if velocity.length > speed:
                new_force = velocity / velocity.length * speed
                person.body._set_velocity(new_force)

        # Remove people within 15 pixels of the exit
        people_to_remove = [(person,speed) for person,speed in self.people if np.linalg.norm(
            np.asarray(person.body.position) - self.EXIT) < 15]
        for person, speed in people_to_remove:
            self._space.remove(person, person.body)
            self.people.remove((person, speed))

        # Add to history
        self.history.append(len(self.people))

        # stop running if no people left
        if len(self.people) == 0:
            self._running = False

    def init_people(self) -> None:
        """
        Create a grid of people
        :return: None
        """
        for i in range(20):
            for j in range(20):
                self.create_person(50 + i*20, 50 + j*20, random.randint(5, 10))

    def create_person(self, x, y, r=5):
        """
        Create a person
        :return: None
        """
        mass = 1
        inertia = pymunk.moment_for_circle(mass, 0, r, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = x, y
        shape = pymunk.Circle(body, r, (0, 0))
        shape.elasticity = 0.95
        shape.friction = 0.0
        self._space.add(body, shape)
        speed = random.randint(50, 150)
        self.people.append((shape, speed))


    ### INTERNALS

    def _run(self):

        # Main loop
        while self._running:
            # Progress time forward
            for x in range(self._physics_steps_per_frame):
                self._space.step(self._dt)

            self._process_events()
            self.update_people()
            self._clear_screen()
            self._draw_objects()
            pygame.display.flip()
            # Delay fixed time between frames
            self._clock.tick(50)
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()))

        return self.history

    def _clear_screen(self) -> None:
        """
        Clears the screen.
        :return: None
        """
        self._screen.fill(pygame.Color("white"))

    def _draw_objects(self) -> None:
        """
        Draw the objects.
        :return: None
        """
        self._space.debug_draw(self._draw_options)

    def _init_world(self):
        # Space
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 0.0)

        # Physics
        # Time step
        self._dt = 1.0 / 60.0
        # Number of physics steps per screen frame
        self._physics_steps_per_frame = 1

        # pygame
        pygame.init()
        self._screen = pygame.display.set_mode((600, 600))
        self._clock = pygame.time.Clock()
        self._draw_options = pymunk.pygame_util.DrawOptions(self._screen)
        
        # Execution control and time until the next ball spawns
        self._running = True

        # Balls that exist in the world
        self.people: List[pymunk.Circle] = []


    def _process_events(self) -> None:
        """
        Handle game and events like keyboard input. Call once per frame only.
        :return: None
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self._running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                pygame.image.save(self._screen, "bouncing_balls.png")


def sim():
    sim = CrowdSim()
    return sim._run()

def main():
    hists = [sim() for _ in range(5)]

    for hist in hists:
        plt.plot(hist)
    
    plt.show()

if __name__ == "__main__":
    main()
