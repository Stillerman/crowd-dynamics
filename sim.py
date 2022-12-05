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

class Person:
    def __init__(self, x, y, r, _space, speed = 100):
        self._space = _space
        self.mass = 1
        inertia = pymunk.moment_for_circle(self.mass, 0, r, (0, 0))
        self.speed = speed
        self.body = pymunk.Body(self.mass, inertia)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, r, (0, 0))
        self.shape.elasticity = 0.95
        self.shape.friction = 0.0
        self._space.add(self.body, self.shape)

        self.panicked = False

        # self.people.append((self.shape, speed))

    def update(self, people, exits):
        if not self.panicked:
            if random.random() < 0.01:
                self.panicked = True

            point_of_interest = np.asarray([250,250])

        if self.panicked:
            nearest_exit = min(exits, key=lambda x: np.linalg.norm(np.array(x) - np.array(self.body.position)))
            point_of_interest = np.asarray(nearest_exit)


        towards_poi = (np.asarray(point_of_interest) -
                        np.asarray(self.body.position))
        towards_poi = towards_poi / np.linalg.norm(towards_poi)
        towards_poi = towards_poi * 500
        self.body._set_force(tuple(towards_poi))

        velocity = self.body._get_velocity()
        if velocity.length > self.speed:
            new_force = velocity / velocity.length * self.speed
            self.body._set_velocity(new_force)

        # Remove people within 15 pixels of any exit
        dist_to_exit = min([np.linalg.norm(np.asarray(self.body.position) - exit) for exit in exits])

        if dist_to_exit < 15:
            self._space.remove(self.shape, self.body)
            people.remove(self)

class CrowdSim(object):

    def __init__(self, MAX_VEL=100, variable_speed=False, jiggle=False) -> None:

        # Params
        self.variable_speed = variable_speed
        self.jiggle = jiggle
        self.MAX_VEL = MAX_VEL
        self.WALL_LENGTH = 500
        self.OFFSET = 10
        self.offset = self.OFFSET
        
        self.exits = []
        # self.exits.append((self.OFFSET + self.WALL_LENGTH / 2, self.OFFSET + 0.95 * self.WALL_LENGTH))
        # self.exits.append((self.OFFSET + self.WALL_LENGTH / 2, self.OFFSET + 0.05 * self.WALL_LENGTH))
        self.exits.append((self.offset, self.offset))
        self.exits.append((self.WALL_LENGTH - self.offset, self.offset))
        self.exits.append((self.offset, self.WALL_LENGTH - self.offset))
        self.exits.append((self.WALL_LENGTH - self.offset, self.WALL_LENGTH - self.offset))

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
        for person in self.people:
            person.update(self.people, self.exits)

        # Add to history
        self.history.append(len(self.people))

        # stop running if no people left
        print(list(map(lambda p: p.body.position, self.people)))
        if len(self.people) == 0:
            self._running = False

    def init_people(self) -> None:
        """
        Create a grid of people
        :return: None
        """
        for i in range(20):
            for j in range(20):
                p = Person(
                    x=50 + i*20,
                    y=50 + j*20,
                    r=5,
                    _space=self._space,
                    speed=self.MAX_VEL
                    )
                self.people.append(p)

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
        self.people: List[Person] = []


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


def sim(varible_speed=False, jiggle=False):
    sim = CrowdSim(variable_speed=varible_speed, jiggle=jiggle)
    return sim._run()

def main():
    hists_vs = [sim(jiggle=True) for _ in range(2)]
    hists_nvs = [sim(jiggle=False) for _ in range(2)]


    for hist in hists_vs:
        plt.plot(hist, label="jiggle")

    for hist in hists_nvs:
        plt.plot(hist, label="no jiggle")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
