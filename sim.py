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
    def __init__(self, x, y, _space, speed = 100, handicapped = False, allow_all_exits = False):
        r = 7 if handicapped else 5
        self._space = _space
        self.allow_all_exits = allow_all_exits
        self.handicapped = handicapped
        self.mass = r**2
        inertia = pymunk.moment_for_circle(self.mass, 0, r, (0, 0))
        self.speed = speed + random.randint(-15, 15)
        self.body = pymunk.Body(self.mass, inertia)
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, r, (0, 0))
        self.shape.elasticity = 0.95
        self.shape.friction = 0.0

        self._space.add(self.body, self.shape)

        self.panicked = False

    def set_color(self):
        if self.handicapped:
            self.shape.color = (255, 0, 0, 255)
        else:
            self.shape.color = (0, 0, 255, 255)
        
    def update(self, exits, handicapped_exits):
        self.set_color()

        if self.handicapped:
            available_exits = [*handicapped_exits]
        elif self.allow_all_exits:
            available_exits = [*handicapped_exits, *exits]
        else:
            available_exits = [*exits]
        # if not self.panicked:
        #     if random.random() < 0.1:
        #         self.panicked = True

        #     point_of_interest = np.asarray([250,250])

        # if self.panicked:

        nearest_exit = min(available_exits, key=lambda x: np.linalg.norm(x - np.array(self.body.position)))
        
        point_of_interest = np.asarray(nearest_exit)

        towards_poi = (np.asarray(point_of_interest) -
                        np.asarray(self.body.position))
        towards_poi = towards_poi / np.linalg.norm(towards_poi)
        towards_poi = towards_poi * 5000 * self.mass
        self.body._set_force(tuple(towards_poi))

        velocity = self.body._get_velocity()
        if velocity.length > self.speed:
            new_force = velocity / velocity.length * self.speed
            self.body._set_velocity(new_force)

        # Remove people within 15 pixels of any exit

        dist_to_exit = min([np.linalg.norm(np.asarray(self.body.position) - exit) for exit in available_exits])

        if dist_to_exit < 30:
            self._space.remove(self.shape, self.body)
            return False

        return True

class CrowdSim(object):

    def __init__(self, MAX_VEL=100, variable_speed=False, jiggle=False, allow_all_exits=False) -> None:

        # Params
        self.variable_speed = variable_speed
        self.allow_all_exits = allow_all_exits
        self.jiggle = jiggle
        self.MAX_VEL = MAX_VEL
        self.WALL_LENGTH = 500
        self.OFFSET = 10
        self.offset = self.OFFSET
        
        self.exits = []
        self.handicapped_exits = []
        # self.exits.append((self.OFFSET + self.WALL_LENGTH / 2, self.OFFSET + 0.95 * self.WALL_LENGTH))
        # self.exits.append((self.OFFSET + self.WALL_LENGTH / 2, self.OFFSET + 0.05 * self.WALL_LENGTH))
        self.exits.append(np.asarray((self.offset, self.offset)))
        self.exits.append(np.asarray((self.WALL_LENGTH - self.offset, self.offset)))
        self.handicapped_exits.append(np.asarray((self.offset, self.WALL_LENGTH - self.offset)))
        self.handicapped_exits.append(np.asarray((self.WALL_LENGTH - self.offset, self.WALL_LENGTH - self.offset)))

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
            if not person.update(self.exits, self.handicapped_exits):
                self.people.remove(person)

        # Add to history
        self.history.append(len(self.people))

        # stop running if no people left
        if len(self.people) <=  2:
            print(list(map(lambda p: p.body.position, self.people)))
        if len(self.people) == 0:
            self._running = False

    def init_people(self) -> None:
        """
        Create a grid of people
        :return: None
        """
        # arrange 20 by 20 people in a grid
        for i in range(20):
            for j in range(20):
                x = self.OFFSET + (self.WALL_LENGTH / 20) * i
                y = self.OFFSET + (self.WALL_LENGTH / 20) * j
                self.people.append(
                    Person(
                        x,
                        y,
                        _space=self._space,
                        handicapped=random.random() < 0.25,
                        speed = 100,
                        allow_all_exits=self.allow_all_exits,
                    )
                )


        # for i in range(20):
        #     for j in range(20):
        #         p = Person(
        #             x=50 + i*15,
        #             y=50 + j*15,
        #             r=5,
        #             _space=self._space,
        #             speed=self.MAX_VEL,
        #             handicapped=random.random() < 0.25
        #             )
        #         self.people.append(p)

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
            pygame.display.set_caption("fps: " + str(self._clock.get_fps()) + " people: " + str(len(self.people)))

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


def sim(varible_speed=False, jiggle=False, allow_all_exits=False):
    sim = CrowdSim(variable_speed=varible_speed, jiggle=jiggle, allow_all_exits=allow_all_exits)
    return sim._run()

def main():
    hists_all = [sim(allow_all_exits=True) for _ in range(2)]
    hists_nall = [sim(allow_all_exits=False) for _ in range(2)]

    for hist in hists_all:
        plt.plot(hist, label="All Exits")

    for hist in hists_nall:
        plt.plot(hist, label="Not All Exits")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
