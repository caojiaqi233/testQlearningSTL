from typing import Optional

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path


class Car2D(gym.Env):

    # metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self):
               
        # defined by cao
        """
        state:                   x,     y,   v_x,   v_y
        state lower bound:       0,     0,  -1.2,  -1.2
        state upper bound:       6,     6,   1.2,   1.2

        action:                a_x,   a_y
        action lower bound:      0,     0
        action upper bound:   0.08,  0.08

        rectangle obstacle:   x_min y_min x_max y_max

        circle goal:          x_goal, y_goal, radius          
        """
        self.map_xmax = 6.0     # range of map
        self.map_xmin = 0.0
        self.map_ymax = 6.0
        self.map_ymin = 0.0
        #self.a = 0.1          # acceleration m/s
        self.tau = 1        # time step s
        self.ob_xmin = 2.0      # range of rectangle obstacle
        self.ob_xmax = 3.0
        self.ob_ymin = 3.0
        self.ob_ymax = 4.0
        self.goal = np.array([5.0,5.0])
        self.goal_r = 0.5

        obs_high = np.array([6.0, 6.0, 1.2, 1.2], dtype=np.float32)
        obs_low = np.array([0.0, 0.0, 0, 0], dtype=np.float32)
        act_high = np.array([0.05, 0.05], dtype=np.float32)
        act_low = np.array([-0.05, -0.05], dtype=np.float32)
        self.action_space = spaces.Box(low=act_low, high=act_high, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_low, high=obs_high, dtype=np.float32)

        """
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0
        self.viewer = None
        
        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        """

    def step(self, a):
        # s:state, a:action, r:reward, d:isdone

        # next step
        s_temp = np.array([0.0,0.0,0.0,0.0])
        
        s_temp[2] = self.s[2]+a[0]*self.tau
        s_temp[2] = np.clip(s_temp[2], self.observation_space.low[2], self.observation_space.high[2])
        
        s_temp[3] = self.s[3]+a[1]*self.tau
        s_temp[3] = np.clip(s_temp[3], self.observation_space.low[3], self.observation_space.high[3])
        
        s_temp[0] = self.s[0]+1/2*(s_temp[2]+self.s[2])*self.tau
        
        s_temp[1] = self.s[1]+1/2*(s_temp[3]+self.s[3])*self.tau
        
        
        # reward and isdone
        d = 0
        if s_temp[0]>=self.ob_xmin and s_temp[0]<=self.ob_xmax and s_temp[1]>=self.ob_ymin and s_temp[1]<=self.ob_ymax:
            r = -100.0
            d = 1
        elif s_temp[0]>self.map_xmax or s_temp[0]<self.map_xmin or s_temp[1]>self.map_ymax or s_temp[1]<self.map_ymin:
            r = -100.0
            d = 1
            """
        elif abs(s_temp[0]-self.goal[0])<=self.goal_r and abs(s_temp[1]-self.goal[1])<=self.goal_r:
            r = 100.0
            d = 0
            """
            """
        elif abs(s_temp[0]-self.goal[0])<=self.goal_r or abs(s_temp[1]-self.goal[1])<=self.goal_r:
            r = 20.0
            d = 0
            """
        else:
            #r = 0.0
            r = -0.01*((s_temp[0]-self.goal[0])**2+(s_temp[1]-self.goal[1])**2)
            d = 0
        i = 'None'      # information
        
        self.s = s_temp

        return(s_temp,r,d,i)
    """
        th, thdot = self.state  # th := theta

        g = self.g
        m = self.m
        l = self.l
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u  # for rendering
        costs = angle_normalize(th) ** 2 + 0.1 * thdot ** 2 + 0.001 * (u ** 2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l ** 2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}
    """
        

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        super().reset(seed=seed)
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()
        """

        self.s = np.array([0.0, 0.0, 0.0, 0.0])
        return self.s


    def render(self):
        # print position and velocity
        print(self.s)    


"""
    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    def render(self, mode="human"):
        if self.viewer is None:
            from gym.utils import pyglet_rendering

            self.viewer = pyglet_rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = pyglet_rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = pyglet_rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = pyglet_rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = pyglet_rendering.Image(fname, 1.0, 1.0)
            self.imgtrans = pyglet_rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u is not None:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None



def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi
"""
